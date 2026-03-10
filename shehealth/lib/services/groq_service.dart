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

      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'model': 'llama-3.3-70b-versatile',
          'messages': messages,
          'temperature': 0.7,
          'max_tokens': 500,
          'top_p': 0.9,
        }),
      );

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

      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'model': 'llama-3.3-70b-versatile',
          'messages': messages,
          'temperature': 0.7,
          'max_tokens': 400,
        }),
      );

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
