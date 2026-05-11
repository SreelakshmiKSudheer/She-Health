import 'package:flutter/material.dart';
import 'dart:convert';
import 'report.dart';
import 'symptom_update_page.dart';
import 'chatbot.dart' show HealthChatbotPage;
import 'calendar.dart';
import 'dietplan.dart';
import 'survey.dart';
import 'services/groq_service.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'personal_details.dart';
import 'settings.dart';
import 'models/app_models.dart';
import 'services/backend_api_service.dart';
import 'services/local_storage_service.dart';
import 'services/session_service.dart';
import 'services/notification_service.dart';
import 'package:provider/provider.dart';
import 'state/app_state.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final ScrollController _scrollController = ScrollController();
  final GroqService _groqService = GroqService();
  final BackendApiService _backendApi = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;
  final SessionService _sessionService = SessionService();

  List<Map<String, dynamic>> _llmReminders = [];
  bool _isReminderLoading = true;
  bool _tipNotificationSent = false;
  bool _reminderNotificationSent = false;
  String _dailyTip = "Loading today's health tip...";
  bool _isTipLoading = true;
  bool _isDashboardLoading = true;

  // Period card state
  String _nextPeriodDateText = '--';
  String _daysUntilPeriodText = '--';

  LocalUserProfile? _localUser;
  Map<String, dynamic>? _latestPrediction;

  int _selectedIndex = 0;

  // Keys for sections to scroll to
  final GlobalKey _nextPeriodKey = GlobalKey();
  final GlobalKey _healthTrendsKey = GlobalKey();
  final GlobalKey _riskAssessmentKey = GlobalKey();
  final GlobalKey _remindersKey = GlobalKey();

  String? _expandedSection;

  // Track selected trend tab
  String _selectedTrendTab = 'Week';

  void _scrollToSection(GlobalKey key) {
    final context = key.currentContext;
    if (context != null) {
      Scrollable.ensureVisible(
        context,
        duration: const Duration(milliseconds: 500),
        curve: Curves.easeInOut,
      );
      Navigator.pop(this.context); // Close drawer
    }
  }

  void _onBottomNavTap(int index) async {
    if (index == 0) {
      setState(() {
        _selectedIndex = 0;
      });
    }

    if (index == 1) {
      await _openLatestReport();
    }

    if (index == 2) {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => const PeriodCalendarWidget(),
        ),
      );
      // Reload period data so the dashboard card refreshes
      if (mounted) {
        await _loadDashboardData();
      }
    }

    if (index == 3) {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => const SurveyPage(),
        ),
      );
    }

    // Reset to home when returning
    setState(() {
      _selectedIndex = 0;
    });
  }

  void _openDietPlan() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const DietPlanPage(),
      ),
    );
  }

  Future<void> _loadDashboardData() async {
    try {
      final userId = await _sessionService.getCurrentUserId();
      if (userId == null) {
        if (!mounted) {
          return;
        }
        setState(() {
          _isDashboardLoading = false;
        });
        return;
      }

      final localUser = await _localStorage.findByUserId(userId);

      Map<String, dynamic>? prediction;
      try {
        prediction = await _backendApi.getLatestPrediction(userId);
      } catch (_) {
        prediction = null;
      }

      // Load period data and compute next period
      String nextPeriodText = '--';
      String daysUntilText = '--';
      try {
        final prefs = await SharedPreferences.getInstance();
        final raw = prefs.getString(kPeriodDataPrefsKey);
        if (raw != null) {
          final data = jsonDecode(raw) as Map<String, dynamic>;
          final year = data['year'] as int?;
          final month = data['month'] as int?;
          final rawDays = data['days'];
          final days = rawDays is List
              ? rawDays
                  .map((e) {
                    if (e is num) {
                      return e.toInt();
                    }
                    if (e is String) {
                      return int.tryParse(e);
                    }
                    return null;
                  })
                  .whereType<int>()
                  .toList()
              : <int>[];
          if (year != null && month != null && days.isNotEmpty) {
            final lastDayOfMonth = DateTime(year, month + 1, 0).day;
            final validDays = days
                .where((d) => d >= 1 && d <= lastDayOfMonth)
                .toSet()
                .toList()
              ..sort();
            if (validDays.isEmpty) {
              throw Exception('No valid saved period days');
            }

            final periodStart = DateTime(year, month, validDays.first);
            final today = DateTime.now();
            final todayDate = DateTime(today.year, today.month, today.day);

            var nextPeriod = periodStart.add(const Duration(days: 28));
            while (nextPeriod.isBefore(todayDate)) {
              nextPeriod = nextPeriod.add(const Duration(days: 28));
            }

            final diff = nextPeriod.difference(todayDate).inDays;
            const monthNames = [
              'Jan',
              'Feb',
              'Mar',
              'Apr',
              'May',
              'Jun',
              'Jul',
              'Aug',
              'Sep',
              'Oct',
              'Nov',
              'Dec'
            ];
            nextPeriodText =
                '${monthNames[nextPeriod.month - 1]} ${nextPeriod.day}, ${nextPeriod.year}';
            daysUntilText = diff > 0 ? '$diff days' : 'Today';
          }
        }
      } catch (_) {}

      if (!mounted) {
        return;
      }

      setState(() {
        _localUser = localUser;
        _latestPrediction = prediction;
        _isDashboardLoading = false;
        _nextPeriodDateText = nextPeriodText;
        _daysUntilPeriodText = daysUntilText;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isDashboardLoading = false;
      });
    }
  }

  Future<void> _openLatestReport() async {
    final userId =
        _localUser?.userId ?? await _sessionService.getCurrentUserId();
    if (!mounted) {
      return;
    }

    String? llmReport;
    if (_latestPrediction != null) {
      llmReport = await _generateLlmReportForPrediction(_latestPrediction!);
    }

    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => HealthReportPage(
          userId: userId,
          predictionData: _latestPrediction,
          localUser: _localUser,
          reportText: _latestPrediction == null
              ? 'No report available yet. Complete the questionnaire to generate your first assessment.'
              : llmReport,
        ),
      ),
    );
  }
  
  String _cleanMarkdown(String text) {
    return text
        .replaceAll('**', '')
        .replaceAll('*', '')
        .trim();
  }

  Future<String?> _generateLlmReportForPrediction(
    Map<String, dynamic> prediction,
  ) async {
    try {
      final raw = prediction['predictions'];
      if (raw is! Map<String, dynamic> || raw.isEmpty) {
        return null;
      }

      final ranked = raw.entries.map((entry) {
        final value = entry.value is Map<String, dynamic>
            ? entry.value as Map<String, dynamic>
            : <String, dynamic>{};
        final probability = (value['probability'] as num? ?? 0).toDouble();
        final label = value['label'] as String? ?? 'Unknown';
        return {
          'condition': entry.key,
          'probability': probability,
          'label': label,
        };
      }).toList()
        ..sort((a, b) => ((b['probability'] as double)
            .compareTo(a['probability'] as double)));

      final top = ranked.take(3).map((item) {
        final probability = (item['probability'] as double).toStringAsFixed(2);
        return '${item['condition']}: $probability (${item['label']})';
      }).join('; ');

      final prompt =
          '''Create a detailed women's health report from model predictions.
    Write 220-320 words in clear, supportive language.
    Use these exact section headers:
    Summary:
    What the risk scores mean:
    Possible contributing factors from current profile:
    Action plan for next 2 weeks:
    When to seek medical review:
    Medical disclaimer:
    Under "What the risk scores mean", explain top 3 risks with one line each.
    Avoid diagnosis and avoid fear-based language.
    Prediction summary: $top.''';

      final result = await _groqService.sendSimpleMessage(prompt);
      final cleaned = _cleanMarkdown(result);
      if (cleaned.isEmpty ||
          cleaned
              .startsWith('I apologize, but I\'m having trouble connecting')) {
        return _buildDetailedFallbackReport(ranked);
      }
      return cleaned;
    } catch (_) {
      return _buildDetailedFallbackReport(const <Map<String, dynamic>>[]);
    }
  }

  String _buildDetailedFallbackReport(List<Map<String, dynamic>> ranked) {
    final top = ranked.take(3).toList();
    String topLine(int index) {
      if (index >= top.length) {
        return '- Not enough scored conditions available yet.';
      }
      final item = top[index];
      final condition = item['condition'] as String? ?? 'Condition';
      final probability = (item['probability'] as num? ?? 0).toDouble();
      final label = item['label'] as String? ?? 'Unknown';
      return '- $condition: score ${probability.toStringAsFixed(2)} ($label). This score shows relative monitoring priority and is not a diagnosis.';
    }

    return '''Summary:
Your current assessment indicates areas to monitor more closely while continuing daily preventive care. These results are intended to support planning and early action.

What the risk scores mean:
${topLine(0)}
${topLine(1)}
${topLine(2)}

Possible contributing factors from current profile:
- Symptoms and lifestyle patterns can influence relative risk scores.
- Incomplete or changing symptom history can shift scores over time.
- Regular tracking improves the reliability of trend interpretation.

Action plan for next 2 weeks:
- Record cycle details, pain level, mood, energy, and sleep each day.
- Maintain hydration, balanced meals, and consistent rest schedule.
- Include regular light exercise and stress-management practices.
- Review this summary with a healthcare professional for personalized advice.

When to seek medical review:
- Seek timely review if symptoms become more frequent, severe, or persistent.
- Seek urgent care for severe pain, heavy bleeding, fainting, or sudden worsening.

Medical disclaimer:
This report is a screening-oriented interpretation and not a clinical diagnosis. A qualified healthcare professional should guide testing and treatment decisions.''';
  }


  Future<void> _generateReminders() async {
  try {
    final predictions = _latestPrediction?['predictions'];

    String conditionSummary = "";

    if (predictions is Map<String, dynamic>) {
      predictions.forEach((key, value) {
        if (value is Map<String, dynamic>) {
          final prob = value['probability'] ?? 0;
          if (prob > 30) {
            conditionSummary += "$key, ";
          }
        }
      });
    }

    final prompt = '''
Generate 4 personalized daily health reminders for a woman.

User:
- Risk Level: $_dashboardRiskLabel
- Conditions: $conditionSummary

Format:
Return ONLY JSON like this:
[
  {"title":"...", "subtitle":"...", "icon":"water"},
  {"title":"...", "subtitle":"...", "icon":"food"}
]

Rules:
- Practical daily actions
- Include diet, hydration, activity, or medication
- Keep each short (under 10 words)
''';

    final response = await _groqService.sendMessage(prompt, []);

    final cleaned = response.trim();

    final parsed = jsonDecode(cleaned);

    setState(() {
      _llmReminders = List<Map<String, dynamic>>.from(parsed);
      _isReminderLoading = false;
    });
    if (!_reminderNotificationSent && _llmReminders.isNotEmpty) {
  await NotificationService.scheduleReminder(
    "${_llmReminders[0]['title']} - ${_llmReminders[0]['subtitle']}"
  );
  _reminderNotificationSent = true;
}
  } catch (e) {
    setState(() {
      _isReminderLoading = false;
      _llmReminders = [
        {
          "title": "Stay Hydrated",
          "subtitle": "Drink enough water",
          "icon": "water"
        }
      ];
    });
    await NotificationService.scheduleReminder(
  _llmReminders[0]['title'],
);
  }
}
  Future<void> _fetchDailyTip() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      String today = DateTime.now().toString().substring(0, 10);

      String? savedDate = prefs.getString('tip_date');
      String? savedTip = prefs.getString('daily_tip');

      // If tip already generated today
      if (savedDate == today && savedTip != null) {
        setState(() {
          _dailyTip = savedTip;
          _isTipLoading = false;
        });
        return;
      }

      // Generate new tip from Groq
      String prompt = '''
Give ONE personalized women's health tip.

User details:
- Risk level: $_dashboardRiskLabel
- Conditions: ${_latestPrediction?.keys.join(", ") ?? "General health"}

Rules:
- Max 25 words
- Actionable
- Friendly tone
- Focus on diet, hormones, or lifestyle
''';

      String response = await _groqService.sendMessage(prompt, []);

      // Save tip and date
      await prefs.setString('daily_tip', response);
      await prefs.setString('tip_date', today);

      setState(() {
        _dailyTip = response;
        _isTipLoading = false;
      });
      if (!_tipNotificationSent) {
  await NotificationService.scheduleHealthTip(_dailyTip);
  _tipNotificationSent = true;
}
    } catch (e) {
      setState(() {
        _dailyTip = "Drink enough water and maintain a healthy routine.";
        _isTipLoading = false;
      });
      if (!_tipNotificationSent) {
  await NotificationService.scheduleHealthTip(_dailyTip);
  _tipNotificationSent = true;
}
    }
  }

  Future<void> _openHealthArticle() async {
    final Uri url =
        Uri.parse("https://www.google.com/search?q=women+health+tips+daily");

    if (await canLaunchUrl(url)) {
      await launchUrl(url, mode: LaunchMode.externalApplication);
    }
  }

  IconData _getIcon(String? icon) {
  switch (icon) {
    case "water":
      return Icons.water_drop;
    case "food":
      return Icons.apple;
    case "walk":
      return Icons.directions_walk;
    case "medicine":
      return Icons.medical_services;
    default:
      return Icons.favorite;
  }
}

  @override
void initState() {
  super.initState();
  _loadDashboardData().then((_) {
    _generateReminders(); // AFTER user data loads
  });
  _fetchDailyTip();
}

  @override
Widget build(BuildContext context) {
  final tip = Provider.of<AppState>(context).healthTip;
  final reminder = Provider.of<AppState>(context).reminderText;
  return Scaffold(
    key: _scaffoldKey,
    backgroundColor: const Color(0xFFFDF2F8),
    drawer: _buildDrawer(),

    body: Column(
      children: [
        _buildHeader(),
        Expanded(
          child: Stack(
            children: [
              SingleChildScrollView(
                controller: _scrollController,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      _buildWelcomeSection(),
                      const SizedBox(height: 20),
                      _buildHealthStatusCards(),
                      const SizedBox(height: 20),
                      _buildMainContent(),
                      const SizedBox(height: 20),
                      _buildHealthTipBanner(),
                      const SizedBox(height: 80),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    ),

    // ✅ ADD FLOATING BUTTONS HERE (CORRECT WAY)
    floatingActionButton: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // 🧪 Test Notification Button
        FloatingActionButton(
          heroTag: 'testNotification',
          onPressed: () async {
            print("🔔 Button pressed");

            await NotificationService.showInstantNotification(
              "Test Notification",
              "This is working ✅",
            );

            print("✅ Notification call done");
          },
          backgroundColor: Colors.green,
          child: const Icon(Icons.notifications),
        ),

        const SizedBox(height: 12),

        // 💬 Chat Button
        FloatingActionButton(
          heroTag: 'chatAI',
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => const HealthChatbotPage(),
              ),
            );
          },
          backgroundColor: const Color(0xFFC85A7A),
          child: const Icon(Icons.chat_bubble),
        ),
      ],
    ),

    bottomNavigationBar: _buildBottomNavigationBar(),
  );
}

  Widget _buildWelcomeSection() {
    return Stack(
      children: [
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [
                Color(0xFFC85A7A),
                Color(0xFFE59393),
                Color.fromARGB(255, 255, 225, 225)
              ],
            ),
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFFE59393).withOpacity(0.3),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Welcome back, ${(_localUser?.fullName ?? 'User').split(' ').first}! 💗',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'Here\'s your health overview for today',
                style: TextStyle(color: Colors.white70, fontSize: 14),
              ),
              const SizedBox(height: 20),
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    ElevatedButton(
                      onPressed: () {
                        final userId = _localUser?.userId;
                        if (userId == null || userId.isEmpty) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Please sign in to continue.'),
                              backgroundColor: Colors.red,
                            ),
                          );
                          return;
                        }
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) =>
                                SymptomUpdatePage(userId: userId),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: const Color(0xFFE59393),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 18, vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      child: const Text(
                        'Log Symptoms',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(
                      onPressed: () {
                        _openLatestReport();
                      },
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.white,
                        side: const BorderSide(color: Colors.white30),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 18, vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      child: const Text(
                        'View Reports',
                        style: TextStyle(fontSize: 12),
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const DietPlanPage(),
                          ),
                        );
                      },
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.white,
                        side: const BorderSide(color: Colors.white30),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 18, vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      child: const Text(
                        'View Diet Plan',
                        style: TextStyle(fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        Positioned(
          top: 0,
          right: -30,
          child: Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildBottomNavigationBar() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            blurRadius: 20,
            offset: const Offset(0, -5),
          ),
        ],
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(24),
          topRight: Radius.circular(24),
        ),
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(24),
          topRight: Radius.circular(24),
        ),
        child: BottomNavigationBar(
          currentIndex: _selectedIndex,
          onTap: _onBottomNavTap,
          type: BottomNavigationBarType.fixed,
          backgroundColor: Colors.white,
          selectedItemColor: const Color(0xFFC85A7A),
          unselectedItemColor: Colors.grey,
          selectedFontSize: 12,
          unselectedFontSize: 12,
          elevation: 0,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home, size: 28),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.description, size: 28),
              label: 'Reports',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.calendar_month, size: 28),
              label: 'Calendar',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.assignment, size: 28),
              label: 'Surveys',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawer() {
    return Drawer(
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(20, 60, 20, 20),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
                      ),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.favorite,
                        color: Colors.white, size: 24),
                  ),
                  const SizedBox(width: 12),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Menu',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      Text(
                        'Quick Navigation',
                        style: TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const Divider(),
            _buildDrawerItem(
                Icons.calendar_today, 'Next Period', _nextPeriodKey),
            _buildDrawerItem(
                Icons.trending_up, 'Health Trends', _healthTrendsKey),
            _buildDrawerItem(
                Icons.monitor_heart, 'Risk Assessment', _riskAssessmentKey),
            _buildDrawerItem(
                Icons.notifications, 'Today\'s Reminders', _remindersKey),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawerItem(IconData icon, String title, GlobalKey key) {
    return ListTile(
      leading: Icon(icon, color: const Color(0xFFE59393)),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
      onTap: () => _scrollToSection(key),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
    );
  }

  Widget _buildHeader() {
    return Stack(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(16, 50, 16, 20),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Color(0xFFC85A7A),
                Color(0xFFE59393),
                Color.fromARGB(255, 255, 225, 225)
              ],
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  GestureDetector(
                    onTap: () => _scaffoldKey.currentState?.openDrawer(),
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 10,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                        child: const Icon(Icons.menu,
                          color: Color(0xFFE59393), size: 28),
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'SHE-HEALTH',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1,
                        ),
                      ),
                      Text(
                        'Women\'s Health Assistance System',
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                ],
              ),
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.settings, color: Colors.white),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SettingsPage(),
                        ),
                      );
                    },
                  ),
                  GestureDetector(
                    onTap: () {
                      final user = _localUser;
                      if (user == null) {
                        return;
                      }
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => PersonalDetailsPage(
                            userId: user.userId,
                            fullName: user.fullName,
                            email: user.email,
                            phone: user.phone,
                            password: user.password,
                            existingProfile: user,
                          ),
                        ),
                      ).then((_) => _loadDashboardData());
                    },
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        shape: BoxShape.circle,
                      ),
                      child: CircleAvatar(
                        radius: 16,
                        backgroundColor: Colors.white,
                        child: Text(
                          _localUser != null
                              ? _localUser!.fullName
                                  .trim()
                                  .split(' ')
                                  .where((e) => e.isNotEmpty)
                                  .map((e) => e[0])
                                  .take(2)
                                  .join()
                                  .toUpperCase()
                              : '?',
                          style: const TextStyle(
                            color: Color(0xFFE59393),
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ),
                  )
                ],
              ),
            ],
          ),
        ),
        Positioned(
          top: 0,
          right: -40,
          child: Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
        Positioned(
          bottom: -20,
          left: -30,
          child: Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildHealthStatusCards() {
    return Column(
      key: _nextPeriodKey,
      children: [
        Row(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFFFCE7F3), width: 2),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.1),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.green.shade50,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.check_circle,
                              color: Colors.green, size: 24),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.green.shade100,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text(
                            'Good',
                            style: TextStyle(
                              color: Colors.green,
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Overall Health',
                      style: TextStyle(color: Colors.grey, fontSize: 14),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _dashboardRiskLabel,
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _latestPrediction == null
                          ? 'Last updated: --'
                          : 'Last updated: Recent',
                      style: const TextStyle(color: Colors.grey, fontSize: 12),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const PeriodCalendarWidget(),
                    ),
                  );
                },
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [
                        Color(0xFFC85A7A),
                        Color(0xFFE59393),
                        Color.fromARGB(255, 255, 225, 225)
                      ],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFFE59393).withOpacity(0.3),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.calendar_today,
                                color: Colors.white, size: 24),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              _daysUntilPeriodText,
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Next Period',
                        style: TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _nextPeriodDateText,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _nextPeriodDateText == '--'
                            ? 'Log a period to predict'
                            : 'Predicted date (+28 days)',
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMainContent() {
    return Column(
      key: _healthTrendsKey,
      children: [
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            SizedBox(
              width: 115,
              child: _buildSectionIcon(
                  'health_trends', Icons.trending_up, 'Health Trends'),
            ),
            SizedBox(
              width: 115,
              child: _buildSectionIcon(
                  'risk_assessment', Icons.monitor_heart, 'Risk Assessment'),
            ),
            SizedBox(
              width: 115,
              child: _buildSectionIcon(
                  'reminders', Icons.notifications, 'Today\'s Reminders'),
            ),
          ],
        ),
        if (_expandedSection != null) ...[
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFFCE7F3), width: 2),
            ),
            child: _buildExpandedContent(),
          ),
        ],
      ],
    );
  }

  Widget _buildSectionIcon(String section, IconData icon, String label) {
    final isActive = _expandedSection == section;

    return InkWell(
      onTap: () {
        setState(() {
          _expandedSection = _expandedSection == section ? null : section;
        });
      },
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isActive ? const Color(0xFFE59393) : Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isActive ? const Color(0xFFE59393) : const Color(0xFFFCE7F3),
            width: 2,
          ),
          boxShadow: isActive
              ? [
                  BoxShadow(
                    color: const Color(0xFFE59393).withOpacity(0.3),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ]
              : null,
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: isActive
                    ? Colors.white.withOpacity(0.2)
                    : const Color(0xFFFCE7F3),
                shape: BoxShape.circle,
              ),
              child: Icon(
                icon,
                color: isActive ? Colors.white : const Color(0xFFE59393),
                size: 28,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: isActive ? Colors.white : Colors.black87,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExpandedContent() {
    switch (_expandedSection) {
      case 'health_trends':
        return _buildHealthTrendsContent();
      case 'risk_assessment':
        return _buildRiskAssessmentContent();
      case 'reminders':
        return _buildRemindersContent();
      default:
        return const SizedBox.shrink();
    }
  }

  // Returns data based on the selected trend tab
  Map<String, Map<String, dynamic>> _getTrendData() {
    switch (_selectedTrendTab) {
      case 'Month':
        return {
          'Symptom Severity': {
            'label': 'Moderate',
            'value': 0.5,
            'color': Colors.orange
          },
          'Stress Level': {'label': 'High', 'value': 0.72, 'color': Colors.red},
          'Energy Level': {
            'label': 'Moderate',
            'value': 0.6,
            'color': Colors.green
          },
          'Mood Score': {'label': 'Fair', 'value': 0.55, 'color': Colors.blue},
          'Weight Changes': {
            'label': '+1.2 kg',
            'value': 0.55,
            'color': Colors.purple
          },
        };
      case 'Year':
        return {
          'Symptom Severity': {
            'label': 'Variable',
            'value': 0.45,
            'color': Colors.deepOrange
          },
          'Stress Level': {
            'label': 'Low',
            'value': 0.3,
            'color': Colors.orange
          },
          'Energy Level': {
            'label': 'Good',
            'value': 0.75,
            'color': Colors.green
          },
          'Mood Score': {'label': 'Great', 'value': 0.85, 'color': Colors.blue},
          'Weight Changes': {
            'label': 'Stable',
            'value': 0.5,
            'color': Colors.purple
          },
        };
      default: // Week
        return {
          'Symptom Severity': {
            'label': 'Low',
            'value': 0.3,
            'color': Colors.pink
          },
          'Stress Level': {
            'label': 'Moderate',
            'value': 0.55,
            'color': Colors.orange
          },
          'Energy Level': {
            'label': 'High',
            'value': 0.8,
            'color': Colors.green
          },
          'Mood Score': {'label': 'Good', 'value': 0.7, 'color': Colors.blue},
          'Weight Changes': {
            'label': 'Stable',
            'value': 0.5,
            'color': Colors.purple
          },
        };
    }
  }

  String _getTrendInsight() {
    switch (_selectedTrendTab) {
      case 'Month':
        return 'Stress levels were elevated mid-month. Consider relaxation techniques to manage better next month.';
      case 'Year':
        return 'Your overall health improved significantly over the year. Mood and energy scores are at an all-time high!';
      default:
        return 'Your symptom severity has decreased by 15% this week. Keep up the healthy habits!';
    }
  }

  String get _dashboardRiskLabel {
    final predictions = _latestPrediction?['predictions'];
    if (predictions is! Map<String, dynamic> || predictions.isEmpty) {
      return _isDashboardLoading ? 'Loading...' : 'No Data';
    }

    double maxProb = 0;
    String maxLabel = 'No Risk';
    for (final item in predictions.values) {
      if (item is! Map<String, dynamic>) {
        continue;
      }
      final p = (item['probability'] as num? ?? 0).toDouble();
      if (p >= maxProb) {
        maxProb = p;
        maxLabel = item['label'] as String? ?? maxLabel;
      }
    }
    return maxLabel;
  }

  List<Map<String, dynamic>> get _predictionRows {
    final predictions = _latestPrediction?['predictions'];
    if (predictions is! Map<String, dynamic>) {
      return const [];
    }

    final rows = <Map<String, dynamic>>[];
    for (final entry in predictions.entries) {
      final value = entry.value;
      if (value is! Map<String, dynamic>) {
        continue;
      }
      final label = value['label'] as String? ?? 'Unknown';
      final prob = (value['probability'] as num? ?? 0).toDouble();

      Color color;
      if (prob < 10) {
        color = Colors.green;
      } else if (prob < 30) {
        color = Colors.lightGreen;
      } else if (prob < 55) {
        color = Colors.orange;
      } else {
        color = Colors.red;
      }

      rows.add({
        'name': entry.key,
        'label': label,
        'color': color,
      });
    }

    return rows;
  }

  Widget _buildHealthTrendsContent() {
    final trendData = _getTrendData();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 10,
          runSpacing: 10,
          alignment: WrapAlignment.spaceBetween,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            const Text(
              'Health Trends',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _buildTabButton('Week'),
                _buildTabButton('Month'),
                _buildTabButton('Year'),
              ],
            ),
          ],
        ),
        const SizedBox(height: 20),
        ...trendData.entries.map((entry) => _buildProgressBar(
              entry.key,
              entry.value['label'] as String,
              entry.value['value'] as double,
              entry.value['color'] as Color,
            )),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFFFCE7F3),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFFFBCFE8)),
          ),
          child: Row(
            children: [
              const Icon(Icons.trending_up, color: Color(0xFFE59393)),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _selectedTrendTab == 'Week'
                          ? 'Positive Trend'
                          : _selectedTrendTab == 'Month'
                              ? 'Monthly Summary'
                              : 'Yearly Overview',
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Color(0xFFC85A7A),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _getTrendInsight(),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildRiskAssessmentContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Risk Assessment',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 20),
        if (_predictionRows.isEmpty)
          const Text('No assessment available. Complete questionnaire first.'),
        ..._predictionRows.map(
          (item) => _buildRiskItem(
            item['name'] as String,
            'Latest',
            item['label'] as String,
            item['color'] as Color,
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: () {
              _openLatestReport();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
              ),
            ),
            child: const Text(
              'Full Assessment',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        )
      ],
    );
  }

  Widget _buildRemindersContent() {
  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      const Text(
        'Today\'s Reminders',
        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
      ),
      const SizedBox(height: 20),

      if (_isReminderLoading)
        const Center(child: CircularProgressIndicator()),

      if (!_isReminderLoading && _llmReminders.isEmpty)
        const Text("No reminders available"),

      ..._llmReminders.map((item) => _buildReminderItem(
            _getIcon(item['icon']),
            item['title'],
            item['subtitle'],
            const Color(0xFFC85A7A),
          )),
    ],
  );
}

  // Updated: tappable tab button that updates _selectedTrendTab
  Widget _buildTabButton(String label) {
    final bool active = _selectedTrendTab == label;
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedTrendTab = label;
        });
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          gradient: active
              ? const LinearGradient(
                  colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
                )
              : null,
          color: active ? null : Colors.grey.shade100,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: active ? Colors.white : Colors.grey.shade600,
            fontWeight: FontWeight.w600,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  Widget _buildProgressBar(
      String label, String value, double progress, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(label,
                    softWrap: true,
                    style: const TextStyle(color: Colors.grey, fontSize: 14)),
              ),
              const SizedBox(width: 10),
              Text(
                value,
                textAlign: TextAlign.right,
                softWrap: true,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: progress,
            backgroundColor: const Color(0xFFFCE7F3),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 8,
            borderRadius: BorderRadius.circular(4),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskItem(String title, String date, String status, Color color) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
            softWrap: true,
          ),
          const SizedBox(height: 4),
          Text(
            'Last checked: $date',
            style: const TextStyle(color: Colors.grey, fontSize: 12),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                status,
                softWrap: true,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReminderItem(
      IconData icon, String title, String subtitle, Color color) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 14),
                  softWrap: true,
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(color: Colors.grey, fontSize: 12),
                  softWrap: true,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHealthTipBanner() {
    return Stack(
      children: [
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFFC85A7A), Color(0xFFE59393), Color(0xFFE59393)],
            ),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFFE59393).withOpacity(0.3),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  shape: BoxShape.circle,
                ),
                child:
                    const Icon(Icons.favorite, color: Colors.white, size: 32),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Daily Health Tip',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isTipLoading
                          ? "Generating today's health tip..."
                          : _dailyTip,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: ElevatedButton(
                        onPressed: _openHealthArticle,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: const Color(0xFFE59393),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(20),
                          ),
                        ),
                        child: const Text(
                          'Learn More',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        Positioned(
          top: -20,
          right: -20,
          child: Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1), shape: BoxShape.circle),
          ),
        ),
      ],
    );
  }
}
