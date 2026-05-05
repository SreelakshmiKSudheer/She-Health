import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;

import '../models/app_models.dart';

class BackendApiService {
  BackendApiService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  String get _baseUrl {
    final configured = dotenv.env['BACKEND_BASE_URL']?.trim();
    if (configured != null && configured.isNotEmpty) {
      return configured.endsWith('/')
          ? configured.substring(0, configured.length - 1)
          : configured;
    }

    if (kIsWeb) {
      final host = Uri.base.host;
      final resolvedHost = host == 'localhost' ? '127.0.0.1' : host;
      return '${Uri.base.scheme}://$resolvedHost:8000';
    }

    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }

    return 'http://localhost:8000';
  }

  Uri _uri(String path) => Uri.parse('$_baseUrl$path');

  Uri _uriForBase(String baseUrl, String path) => Uri.parse('$baseUrl$path');

  List<String> _submitBaseUrlCandidates() {
    final primary = _baseUrl;
    final candidates = <String>[primary];

    if (!kIsWeb) {
      return candidates;
    }

    final parsed = Uri.tryParse(primary);
    if (parsed == null || parsed.host.isEmpty) {
      return candidates;
    }

    void addHostVariant(String host) {
      final normalized = parsed
          .replace(host: host)
          .toString()
          .replaceAll(RegExp(r'/$'), '');
      if (!candidates.contains(normalized)) {
        candidates.add(normalized);
      }
    }

    if (parsed.host == '127.0.0.1') {
      addHostVariant('localhost');
    } else if (parsed.host == 'localhost') {
      addHostVariant('127.0.0.1');
    }

    final browserHost = Uri.base.host;
    if (browserHost == 'localhost' || browserHost == '127.0.0.1') {
      addHostVariant(browserHost);
    }

    return candidates;
  }

  Future<http.Response> _postWithLoopbackFallback(
    String path,
    Map<String, dynamic> payload,
  ) async {
    Object? lastError;
    final candidates = _submitBaseUrlCandidates();

    for (final baseUrl in candidates) {
      try {
        return await _client.post(
          _uriForBase(baseUrl, path),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        );
      } catch (e) {
        lastError = e;
      }
    }

    throw Exception(
      'Unable to reach backend at port 8000. Tried: ${candidates.join(', ')}. '
      'Please ensure the FastAPI backend is running. Last error: $lastError',
    );
  }

  Future<http.Response> _requestWithLoopbackFallback({
    required String method,
    required String path,
    Map<String, dynamic>? payload,
  }) async {
    Object? lastError;
    final candidates = _submitBaseUrlCandidates();

    for (final baseUrl in candidates) {
      try {
        final uri = _uriForBase(baseUrl, path);
        final headers = {'Content-Type': 'application/json'};
        if (method == 'GET') {
          return await _client.get(uri, headers: headers);
        }
        if (method == 'PATCH') {
          return await _client.patch(
            uri,
            headers: headers,
            body: payload == null ? null : jsonEncode(payload),
          );
        }
      } catch (e) {
        lastError = e;
      }
    }

    throw Exception(
      'Unable to reach backend at port 8000. Tried: ${candidates.join(', ')}. '
      'Please ensure the FastAPI backend is running. Last error: $lastError',
    );
  }

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

    if (response.statusCode == 400 &&
        response.body.contains('already registered')) {
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
    final payload = {
      'user_id': userId,
      'responses': selectedOptionIdsByQuestion.entries
          .map(
            (entry) => {
              'question_id': entry.key,
              'selected_option_ids': entry.value,
            },
          )
          .toList(),
    };

    final response = await _postWithLoopbackFallback('/response/submit', payload);

    if (response.statusCode >= 400) {
      throw Exception('Failed to submit responses: ${response.body}');
    }
  }

  Future<void> updateResponses({
    required String userId,
    required Map<String, List<String>> selectedOptionIdsByQuestion,
  }) async {
    final payload = {
      'user_id': userId,
      'responses': selectedOptionIdsByQuestion.entries
          .map(
            (entry) => {
              'question_id': entry.key,
              'selected_option_ids': entry.value,
            },
          )
          .toList(),
    };

    final response = await _requestWithLoopbackFallback(
      method: 'PATCH',
      path: '/response/update',
      payload: payload,
    );

    if (response.statusCode >= 400) {
      throw Exception('Failed to update responses: ${response.body}');
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
    final response = await _requestWithLoopbackFallback(
      method: 'GET',
      path: '/response/$userId',
    );
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch user responses: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}
