import 'package:shared_preferences/shared_preferences.dart';

class SessionService {
  static const _currentUserKey = 'current_user_id';
  static const _ageKey = 'age';
  static const _conditionKey = 'condition';
  static const _symptomsKey = 'symptoms';

  Future<void> setCurrentUserId(String userId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_currentUserKey, userId);
  }

  Future<String?> getCurrentUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_currentUserKey);
  }

  Future<void> clearCurrentUser() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_currentUserKey);
   }

   Future<void> saveUserData({
  required int age,
  required String condition,
  required List<String> symptoms,
}) async {
  final prefs = await SharedPreferences.getInstance();

  await prefs.setInt(_ageKey, age);
  await prefs.setString(_conditionKey, condition);
  await prefs.setStringList(_symptomsKey, symptoms);
}

  Future<Map<String, dynamic>> getUserData() async {
  final prefs = await SharedPreferences.getInstance();

  return {
    "age": prefs.getInt(_ageKey) ?? 0,
    "condition": prefs.getString(_conditionKey) ?? "Unknown",
    "symptoms": prefs.getStringList(_symptomsKey) ?? [],
  };
}
}
