import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi_web/sqflite_ffi_web.dart';
import 'auth_page.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:provider/provider.dart';
import 'firebase_options.dart';
import 'services/notification_service.dart';
import 'state/app_state.dart';
import 'package:shehealth/app_navigator.dart';
import 'dietplan.dart';
import 'shehealth_dashboard.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await NotificationService.init();

  if (kIsWeb) {
    databaseFactory = databaseFactoryFfiWeb;
  }

  await dotenv.load(fileName: ".env");

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp( ChangeNotifierProvider(
    create: (_) => AppState(
      dietPlan: "",
      workoutPlan: "",
      healthTip: "",
      reminderText: "",
    ),
    child: const SheHealthApp(),
  ),
);
}

class SheHealthApp extends StatelessWidget {
  const SheHealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      /// 🧭 GLOBAL NAVIGATION CONTROL
      navigatorKey: navigatorKey,

      title: 'SheHealth',
      debugShowCheckedModeBanner: false,

      theme: ThemeData(
        primaryColor: const Color(0xFFC85A7A),
        scaffoldBackgroundColor: Colors.white,
        fontFamily: 'Poppins',

        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFC85A7A),
          primary: const Color(0xFFC85A7A),
          secondary: const Color(0xFFE59393),
        ),

        useMaterial3: true,

        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFC85A7A),
          foregroundColor: Colors.white,
          elevation: 0,
        ),

        floatingActionButtonTheme: const FloatingActionButtonThemeData(
          backgroundColor: Color(0xFFC85A7A),
          foregroundColor: Colors.white,
        ),

        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFFC85A7A),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(
              horizontal: 24,
              vertical: 12,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),

        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFE5C4C4)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFE5C4C4)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(
              color: Color(0xFFC85A7A),
              width: 2,
            ),
          ),
          filled: true,
          fillColor: const Color(0xFFFFF5F8),
        ),
      ),

      /// 🧭 ROUTES (Navigation mapping for notifications + app)
      routes: {
        '/diet': (context) => DietPlanPage(),
        '/workout': (context) => DietPlanPage(),
        '/tips': (context) => DashboardPage(),
        '/reminders': (context) => DashboardPage(),
      },

      /// 🚀 FIRST SCREEN
      home: const AuthPage(),
    );
  }
}