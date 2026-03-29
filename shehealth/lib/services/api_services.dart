import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {

  static const String baseUrl = "http://10.0.2.2:8000";

  static Future<bool> registerUser(Map<String, dynamic> data) async {

    final response = await http.post(
      Uri.parse("$baseUrl/users/"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(data),
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      return true;
    } else {
      print(response.body);
      return false;
    }
  }
}