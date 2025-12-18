import 'package:flutter/material.dart';
import 'auth_page.dart';
import 'shehealth_dashboard.dart';
import 'questionnaire.dart';
import 'report.dart';

void main() {
  runApp(const SheHealthApp());
}

class SheHealthApp extends StatelessWidget {
  const SheHealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SheHealth',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFFC85A7A),
        scaffoldBackgroundColor: Colors.white,
        fontFamily: 'Poppins',
      ),
      home: const AuthPage(),
    );
  }
}