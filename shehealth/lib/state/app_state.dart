import 'package:flutter/material.dart';

class AppState extends ChangeNotifier {
  String dietPlan;
  String workoutPlan;
  String healthTip;
  String reminderText;

  AppState({
    required this.dietPlan,
    required this.workoutPlan,
    required this.healthTip,
    required this.reminderText,
  });

  // ✅ Optional: update methods
  void updateDiet(String value) {
    dietPlan = value;
    notifyListeners();
  }

  void updateWorkout(String value) {
    workoutPlan = value;
    notifyListeners();
  }

  void updateTip(String value) {
    healthTip = value;
    notifyListeners();
  }

  void updateReminder(String value) {
    reminderText = value;
    notifyListeners();
  }
}