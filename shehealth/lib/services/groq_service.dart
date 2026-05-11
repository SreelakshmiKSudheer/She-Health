import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';

class GroqService {
  static const String baseUrl =
      'https://api.groq.com/openai/v1/chat/completions';

  final String apiKey;

  GroqService({String? apiKey})
      : apiKey = apiKey ?? dotenv.env['GROQ_API_KEY'] ?? '';

  Future<Map<String, dynamic>> _postChatCompletion(
    List<Map<String, dynamic>> messages, {
    double temperature = 0.7,
    int maxTokens = 500,
  }) async {
    final response = await http
        .post(
          Uri.parse(baseUrl),
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $apiKey',
          },
          body: jsonEncode({
            'model': 'llama-3.1-8b-instant',
            'messages': messages,
            'temperature': temperature,
            'max_tokens': maxTokens,
            'top_p': 0.9,
          }),
        )
        .timeout(const Duration(seconds: 60));

    if (response.statusCode != 200) {
      throw Exception(
        'Failed to get response: ${response.statusCode} - ${response.body}',
      );
    }

    final data = jsonDecode(response.body);
    if (data is! Map<String, dynamic>) {
      throw Exception('Unexpected response format from Groq');
    }

    return data;
  }

  Future<String> sendMessage(String userMessage,
      List<Map<String, dynamic>> conversationHistory) async {
    if (apiKey.isEmpty) {
      throw Exception('Groq API key missing');
    }
    try {
      final List<Map<String, dynamic>> messages = [
        {
          'role': 'system',
          'content':
              '''You are a compassionate and knowledgeable women's health assistant specializing in:
- Menstrual health and cycle management
- PCOS/PCOD awareness and management
- Endometriosis information
- Cervical cancer prevention and screening
- Pregnancy and fertility guidance
- General women's wellness

Provide accurate, empathetic, and helpful information. Always remind users to consult healthcare professionals for personalized medical advice, diagnosis, or treatment. Keep responses concise (2-3 paragraphs) and easy to understand.'''
        },
        ...conversationHistory,
        {'role': 'user', 'content': userMessage}
      ];

      final data = await _postChatCompletion(messages);
      final content = data['choices']?[0]?['message']?['content'];
      return content is String ? content : '';
    } catch (e) {
      return 'I apologize, but I\'m having trouble connecting right now. Please try again in a moment.';
    }
  }

  Future<String> sendSimpleMessage(String prompt) async {
    if (apiKey.isEmpty) {
      throw Exception('Groq API key missing');
    }
    try {
      final response = await sendMessage(prompt, []);
      return response;
    } catch (e) {
      return "Unable to generate health tip right now.";
    }
  }

  // Exponential backoff: 2s, 4s, 8s, 16s, 30s
  static int _getBackoffDelaySeconds(int retryCount) {
    if (retryCount == 0) return 2;
    if (retryCount == 1) return 4;
    if (retryCount == 2) return 8;
    if (retryCount == 3) return 16;
    return 30;
  }

  // Try to repair common JSON issues in model output and return a canonical JSON string.
  // Returns null if unable to repair.
  String? _attemptRepairJson(String raw) {
    try {
      var t = raw.trim();

      // Replace smart quotes with ASCII quotes
      t = t.replaceAll('“', '"').replaceAll('”', '"').replaceAll('‘', "'").replaceAll('’', "'");

      // Extract the first JSON-like block (either object or array)
      final startIdx = t.indexOf(RegExp(r'[\[{]'));
      final endIdx = t.lastIndexOf(RegExp(r'[\]}]'));
      if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
        t = t.substring(startIdx, endIdx + 1);
      }

      // Remove trailing commas before closing braces/brackets
      t = t.replaceAllMapped(RegExp(r',\s*([}\]])'), (m) => m.group(1) ?? '');

      // Balance curly braces
      final openCurly = t.split('{').length - 1;
      final closeCurly = t.split('}').length - 1;
      if (openCurly > closeCurly) {
        t = t + List.filled(openCurly - closeCurly, '}').join();
      }

      // Balance square brackets
      final openSquare = t.split('[').length - 1;
      final closeSquare = t.split(']').length - 1;
      if (openSquare > closeSquare) {
        t = t + List.filled(openSquare - closeSquare, ']').join();
      }

      // Final attempt to parse
      final parsed = jsonDecode(t);
      // Return a canonical JSON string so callers get valid JSON
      return jsonEncode(parsed);
    } catch (e) {
      // Give up — return null to let caller decide how to handle
      print('JSON repair failed: $e');
      return null;
    }
  }

  Future<String> sendHealthPlanMessage(
    String prompt, {
    int retryCount = 0,
    int maxTokens = 1200,
  }) async {
    if (apiKey.isEmpty) {
      throw Exception('Groq API key missing');
    }

    try {
      final data = await _postChatCompletion(
        [
          {
            'role': 'system',
            'content':
                'Return ONLY valid JSON. Do not wrap the result in markdown or add any extra text. Do not use code fences.'
          },
          {
            'role': 'user',
            'content': prompt
          }
        ],
        temperature: 0.2,
        maxTokens: maxTokens,
      );

      final content = data['choices']?[0]?['message']?['content'];
      if (content is! String) return '';

      // Validate JSON — try to parse; if invalid, attempt to repair common issues
      try {
        jsonDecode(content);
        return content;
      } catch (_) {
        final repaired = _attemptRepairJson(content);
        // If repair succeeded, return the repaired string; otherwise fallthrough to throw below
        if (repaired != null) return repaired;
        throw Exception('Incomplete JSON');
      }

    } catch (e) {
      final errorText = e.toString();
      if (errorText.contains('429')) {
        final delaySeconds = _getBackoffDelaySeconds(retryCount);
        print('⚠️ Rate limit (429) - waiting ${delaySeconds}s before retry ${retryCount + 1}/5');
        
        await Future.delayed(Duration(seconds: delaySeconds));

        if (retryCount < 4) {
          // Max 5 total attempts (0-4)
          return sendHealthPlanMessage(
            prompt,
            retryCount: retryCount + 1,
          );
        }
        throw Exception('Rate limit exceeded after 5 attempts');
      }

      throw Exception(
        'Health plan generation failed: $e',
      );
    }
  }

  Future<String> generateDietPlan(String condition) async {
    try {
      final List<Map<String, dynamic>> messages = [
        {
          'role': 'system',
          'content': '''You are a certified women's health nutrition assistant.

Create a healthy and balanced daily diet plan specifically for women based on their health condition.

The response should include:
Breakfast
Mid-Morning Snack
Lunch
Evening Snack
Dinner

Keep the food simple, nutritious, and commonly available.'''
        },
        {
          'role': 'user',
          'content': 'Generate a daily diet plan for a woman with $condition.'
        }
      ];

      final data = await _postChatCompletion(
        messages,
        temperature: 1.0,
        maxTokens: 1200,
      );
      final content = data['choices']?[0]?['message']?['content'];
      return content is String ? content : '';
    } catch (e) {
      return 'Unable to generate diet plan right now. Please try again later.';
    }
  }
}
