import 'dart:convert';

import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;

import '../models/app_models.dart';

class BackendApiService {
  BackendApiService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  String get _baseUrl {
    final configured = dotenv.env['BACKEND_BASE_URL']?.trim();
    if (configured != null && configured.isNotEmpty) {
      return configured;
    }
    return 'http://10.0.2.2:8000';
  }

  Uri _uri(String path) => Uri.parse('$_baseUrl$path');

  Future<void> registerUserProfile({
    required String userId,
    required int age,
    required double height,
    required double weight,
    required String? maritalStatus,
    required bool familyHistory,
  }) async {
    final payload = {
      'user_id': userId,
      'age': age,
      'height': height,
      'weight': weight,
      'marital_status': (maritalStatus == null || maritalStatus.isEmpty)
          ? 'single'
          : maritalStatus.toLowerCase(),
      'family_history': familyHistory,
    };

    final response = await _client.post(
      _uri('/users/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    );

    if (response.statusCode < 400) {
      return;
    }

    if (response.statusCode == 400 && response.body.contains('already registered')) {
      final updateResponse = await _client.put(
        _uri('/users/$userId'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'age': age,
          'height': height,
          'weight': weight,
          'marital_status': (maritalStatus == null || maritalStatus.isEmpty)
              ? 'single'
              : maritalStatus.toLowerCase(),
          'family_history': familyHistory,
        }),
      );
      if (updateResponse.statusCode < 400) {
        return;
      }
      throw Exception('Failed to update user profile: ${updateResponse.body}');
    }

    throw Exception('Failed to register user profile: ${response.body}');
  }

  Future<Map<String, dynamic>> getUserProfile(String userId) async {
    final response = await _client.get(_uri('/users/$userId'));
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch user profile: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<QuestionnaireQuestion>> fetchQuestionnaire() async {
    final response = await _client.get(_uri('/questionnaire/'));
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch questionnaire: ${response.body}');
    }

    final list = jsonDecode(response.body) as List<dynamic>;
    return list
        .map((e) => QuestionnaireQuestion.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> submitResponses({
    required String userId,
    required Map<String, List<String>> selectedOptionIdsByQuestion,
  }) async {
    final response = await _client.post(
      _uri('/response/submit'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'user_id': userId,
        'responses': selectedOptionIdsByQuestion.entries
            .map(
              (entry) => {
                'question_id': entry.key,
                'selected_option_ids': entry.value,
              },
            )
            .toList(),
      }),
    );

    if (response.statusCode >= 400) {
      throw Exception('Failed to submit responses: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> runPrediction(String userId) async {
    final response = await _client.post(_uri('/predict/$userId'));
    if (response.statusCode >= 400) {
      throw Exception('Failed to run prediction: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> getLatestPrediction(String userId) async {
    final response = await _client.get(_uri('/predict/latest/$userId'));
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch latest prediction: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> getUserResponses(String userId) async {
    final response = await _client.get(_uri('/response/$userId'));
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch user responses: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}
