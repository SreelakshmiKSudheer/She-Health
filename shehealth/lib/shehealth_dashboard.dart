import 'package:flutter/material.dart';
import 'report.dart';
import 'questionnaire.dart';
import 'chatbot.dart';
import 'calendar.dart';
import 'dietplan.dart';
import 'survey.dart';
import 'services/groq_service.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final ScrollController _scrollController = ScrollController();
  final GroqService _groqService = GroqService();

String _dailyTip = "Loading today's health tip...";
bool _isTipLoading = true;

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
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => HealthReportPage(
          reportText: "No report available.",
        ),
      ),
    );
  }

  if (index == 2) {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const PeriodCalendarWidget(),
      ),
    );
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
    String prompt =
        "Give one short helpful women's health tip for today. Keep it under 30 words.";

    String response = await _groqService.sendMessage(prompt, []);

    // Save tip and date
    await prefs.setString('daily_tip', response);
    await prefs.setString('tip_date', today);

    setState(() {
      _dailyTip = response;
      _isTipLoading = false;
    });
  } catch (e) {
    setState(() {
      _dailyTip = "Drink enough water and maintain a healthy routine.";
      _isTipLoading = false;
    });
  }
}

Future<void> _openHealthArticle() async {
  final Uri url = Uri.parse(
      "https://www.google.com/search?q=women+health+tips+daily");

  if (await canLaunchUrl(url)) {
    await launchUrl(url, mode: LaunchMode.externalApplication);
  }
}

@override
void initState() {
  super.initState();
  _fetchDailyTip();
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: const Color(0xFFFDF2F8),
      drawer: _buildDrawer(),
      body: Stack(
        children: [
          SingleChildScrollView(
            controller: _scrollController,
            child: Column(
              children: [
                _buildHeader(),
                Padding(
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
                      const SizedBox(height: 80), // Extra space for bottom nav
                    ],
                  ),
                ),
              ],
            ),
          ),
          // Floating Chat AI Button
          Positioned(
            right: 16,
            bottom: 90,
            child: FloatingActionButton(
              heroTag: 'chatAI',
              onPressed: () {
                // Navigate to Health Chatbot Page
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const HealthChatbotPage()),
                );
              },
              backgroundColor: const Color(0xFFC85A7A),
              elevation: 8,
              child:
                  const Icon(Icons.chat_bubble, color: Colors.white, size: 28),
            ),
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
              const Text(
                'Welcome back, Sarah! 💗',
                style: TextStyle(
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
              Row(
                children: [
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SymptomQuestionnaire(),
                        ),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFFE59393),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 24, vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    child: const Text(
                      'Log Symptoms',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => HealthReportPage(
                            reportText: "No report available.",
                          ),
                        ),
                      );
                    },
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      side: const BorderSide(color: Colors.white30),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 24, vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    child: const Text('View Report'),
                  ),
                  const SizedBox(width: 12),
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
                          horizontal: 24, vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    child: const Text('View Diet Plan'),
                  ),
                ],
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
                      child: const Icon(Icons.favorite,
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
                        'Women\'s Health Predictive System',
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                ],
              ),
              Row(
                children: [
                  Stack(
                    children: [
                      IconButton(
                        icon: const Icon(Icons.notifications,
                            color: Colors.white),
                        onPressed: () {},
                      ),
                      Positioned(
                        right: 8,
                        top: 8,
                        child: Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Colors.yellow,
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    ],
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      children: [
                        CircleAvatar(
                          radius: 16,
                          backgroundColor: Colors.white,
                          child: const Text(
                            'SA',
                            style: TextStyle(
                              color: Color(0xFFE59393),
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Sarah Anderson',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                                fontSize: 12,
                              ),
                            ),
                            Text(
                              'ID: SH2024001',
                              style: TextStyle(
                                  color: Colors.white70, fontSize: 10),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
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
                    const Text(
                      'No Risk',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Last updated: Today',
                      style: TextStyle(color: Colors.grey, fontSize: 12),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
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
                          child: const Text(
                            '5 days',
                            style: TextStyle(
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
                    const Text(
                      'Oct 17, 2025',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Predicted date',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ],
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
        Row(
          children: [
            Expanded(
                child: _buildSectionIcon(
                    'health_trends', Icons.trending_up, 'Health Trends')),
            const SizedBox(width: 12),
            Expanded(
                child: _buildSectionIcon(
                    'risk_assessment', Icons.monitor_heart, 'Risk Assessment')),
            const SizedBox(width: 12),
            Expanded(
                child: _buildSectionIcon(
                    'reminders', Icons.notifications, 'Today\'s Reminders')),
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

  Widget _buildHealthTrendsContent() {
    final trendData = _getTrendData();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Health Trends',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            Row(
              children: [
                _buildTabButton('Week'),
                const SizedBox(width: 8),
                _buildTabButton('Month'),
                const SizedBox(width: 8),
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
        _buildRiskItem('PCOD/PCOS', '3 days ago', 'Low', Colors.green),
        _buildRiskItem('Thyroid', '1 week ago', 'No Risk', Colors.green),
        _buildRiskItem('Endometriosis', '5 days ago', 'Monitor', Colors.orange),
        _buildRiskItem(
            'Cervical Cancer', '2 weeks ago', 'No Risk', Colors.green),
        const SizedBox(height: 16),
        SizedBox(
  width: double.infinity,
  child: ElevatedButton(
    onPressed: () {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => HealthReportPage(
            reportText: "No report available.",
          ),
        ),
      );
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
        _buildReminderItem(
            Icons.apple, 'Take Vitamin D', 'After breakfast', Colors.pink),
        _buildReminderItem(Icons.water_drop, 'Drink Water - 2L',
            'Stay hydrated throughout the day', Colors.blue),
        _buildReminderItem(Icons.directions_walk, 'Evening Walk',
            '30 minutes recommended', Colors.purple),
        _buildReminderItem(Icons.calendar_today, 'Pap Smear Due',
            'Schedule in 2 weeks', Colors.pink),
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
              Text(label,
                  style: const TextStyle(color: Colors.grey, fontSize: 14)),
              Text(
                value,
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
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 14),
                ),
                const SizedBox(height: 4),
                Text(
                  'Last checked: $date',
                  style: const TextStyle(color: Colors.grey, fontSize: 12),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              status,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
                fontSize: 12,
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
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(color: Colors.grey, fontSize: 12),
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
  _isTipLoading ? "Generating today's health tip..." : _dailyTip,
  style: const TextStyle(
    color: Colors.white70,
    fontSize: 13,
  ),
),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
  onPressed: _openHealthArticle,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: const Color(0xFFE59393),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),
                child: const Text(
                  'Learn More',
                  style: TextStyle(fontWeight: FontWeight.bold),
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
