import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';

class GroqService {
  static const String baseUrl =
      'https://api.groq.com/openai/v1/chat/completions';

  final String apiKey;

  GroqService({String? apiKey})
      : apiKey = apiKey ?? dotenv.env['GROQ_API_KEY'] ?? '';

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
          'temperature': 0.7,
          'max_tokens': 500,
          'top_p': 0.9,
        }),
      )
.timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final content = data['choices']?[0]?['message']?['content'];
        return content is String ? content : '';
      } else {
        throw Exception(
            'Failed to get response: ${response.statusCode} - ${response.body}');
      }
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

  Future<String> sendHealthPlanMessage(
  String prompt, {
  int retryCount = 0,
}) async {

  if (apiKey.isEmpty) {
    throw Exception('Groq API key missing');
  }

  try {

    final response = await http
    .post(
      Uri.parse(baseUrl),

      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      },

      body: jsonEncode({

        // STRONG MODEL
        'model': 'llama-3.1-8b-instant',
        'messages': [
          {
            'role': 'system',
            'content':
                'Return ONLY valid JSON.'
          },

          {
            'role': 'user',
            'content': prompt
          }
        ],

        'temperature': 0.3,

        'max_tokens': 1200,

        'top_p': 0.8,
      }),
    ).timeout(const Duration(seconds: 30));

    if (response.statusCode == 200) {

      final data =
          jsonDecode(response.body);

      final content =
          data['choices']?[0]
              ?['message']
              ?['content'];

      return content is String
          ? content
          : '';

    } else {
      if (response.statusCode == 429) {

  await Future.delayed(
    const Duration(seconds: 2),
  );

  if (retryCount < 3) {

  return sendHealthPlanMessage(
    prompt,
    retryCount: retryCount + 1,
  );
}
throw Exception('Rate limit exceeded');
}
      throw Exception(
        'Groq Error: ${response.body}',
      );
    }

  } catch (e) {

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
          'temperature': 0.7,
          'max_tokens': 400,
        }),
      )
.timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final content = data['choices']?[0]?['message']?['content'];
        return content is String ? content : '';
      } else {
        throw Exception(
            'Failed to generate diet plan: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      return 'Unable to generate diet plan right now. Please try again later.';
    }
  }
}