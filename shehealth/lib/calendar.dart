import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:math' as math;
import 'package:shared_preferences/shared_preferences.dart';

import 'services/backend_api_service.dart';
import 'services/session_service.dart';

/// Shared key so the dashboard can also read saved period data.
const String kPeriodDataPrefsKey = 'period_days_v1';

class PeriodCalendarWidget extends StatefulWidget {
  const PeriodCalendarWidget({super.key});

  @override
  State<PeriodCalendarWidget> createState() => _PeriodCalendarWidgetState();
}

class _PeriodCalendarWidgetState extends State<PeriodCalendarWidget>
    with TickerProviderStateMixin {
  DateTime selectedDate = DateTime.now();
  int selectedTab = 0;

  final BackendApiService _backendApi = BackendApiService();
  final SessionService _sessionService = SessionService();

  String _currentPhase = "Follicular Phase";
  final int _daysUntilPeriod = 5;
  final List<String> _selectedSymptoms = [];

  List<int> periodDays = [1, 2, 3, 4, 5];
  List<int> fertileDays = [13];
  int? selectedDay;

  // Notification state
  bool _notifPeriodReminder = true;
  bool _notifFertileWindow = true;
  bool _notifMedReminder = false;
  bool _notifCycleInsights = true;
  bool _hasUnread = true;

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _loadPeriodData();
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  // ── Period persistence ───────────────────────────────────────────────────

  Future<void> _loadPeriodData() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(kPeriodDataPrefsKey);
      if (raw == null) return;
      final data = jsonDecode(raw) as Map<String, dynamic>;
      final year = data['year'] as int?;
      final month = data['month'] as int?;
      final days = (data['days'] as List?)?.cast<int>() ?? [];
      if (year == null || month == null) return;
      if (!mounted) return;
      setState(() {
        selectedDate = DateTime(year, month, 1);
        periodDays = days;
      });
    } catch (_) {}
  }

  Future<void> _savePeriodData() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        kPeriodDataPrefsKey,
        jsonEncode({
          'year': selectedDate.year,
          'month': selectedDate.month,
          'days': List<int>.from(periodDays),
        }),
      );

      final userId = await _sessionService.getCurrentUserId();
      if (userId != null && userId.isNotEmpty) {
        final day = selectedDay ?? selectedDate.day;
        final dateString = DateTime(selectedDate.year, selectedDate.month, day)
            .toIso8601String()
            .split('T')
            .first;

        try {
          await _backendApi.logCycle(
            userId: userId,
            date: dateString,
            symptoms: List<String>.from(_selectedSymptoms),
            flow: 'normal',
          );
        } catch (_) {
          // optional: handle network failure gracefully, offline mode etc.
        }
      }
    } catch (_) {}
  }

  void _changeMonth(int months) {
    setState(() {
      selectedDate =
          DateTime(selectedDate.year, selectedDate.month + months, 1);
      selectedDay = null;
    });
  }

  void _changeYear(int years) {
    setState(() {
      selectedDate = DateTime(selectedDate.year + years, selectedDate.month, 1);
      selectedDay = null;
    });
  }

  void _updateCycleInsights(int dayOfCycle) {
    setState(() {
      if (dayOfCycle <= 5) {
        _currentPhase = "Menstrual Phase";
        // Update notifications/tips
      } else if (dayOfCycle <= 13) {
        _currentPhase = "Follicular Phase";
      } else if (dayOfCycle <= 16) {
        _currentPhase = "Ovulation Phase";
      } else {
        _currentPhase = "Luteal Phase";
      }
    });
  }

  int _getDaysInMonth([DateTime? date]) {
    date ??= selectedDate;
    return DateTime(date.year, date.month + 1, 0).day;
  }

  String _getMonthName(int month) {
    const months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December'
    ];
    return months[month - 1];
  }

  String _getShortMonthName(int month) {
    const months = [
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
    return months[month - 1];
  }

  String _getDayName(DateTime date) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return days[date.weekday - 1];
  }

  void _showCalendarModal() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      enableDrag: true,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.78,
        minChildSize: 0.45,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => _FullCalendarModal(
          selectedDate: selectedDate,
          periodDays: periodDays,
          fertileDays: fertileDays,
          scrollController: scrollController,
          onDayTap: (day) {
            setState(() {
              selectedDay = day;
              if (periodDays.contains(day)) {
                periodDays.remove(day);
              } else {
                periodDays.add(day);
              }
            });
            _updateCycleInsights(day);
            _savePeriodData();
          },
          onMonthChanged: (date) {
            setState(() {
              selectedDate = date;
              selectedDay = null;
            });
          },
        ),
      ),
    );
  }

  // ── Notification Panel ────────────────────────────────────────────────────

  void _showNotificationPanel() {
    setState(() => _hasUnread = false);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          final List<Map<String, dynamic>> upcoming = [
            {
              'icon': Icons.water_drop_rounded,
              'color': const Color(0xFFC85A7A),
              'title': 'Period expected soon',
              'subtitle': 'Based on your 28-day cycle — around day 28',
              'time': 'In ~3 days',
            },
            {
              'icon': Icons.favorite_rounded,
              'color': const Color(0xFF9B84D4),
              'title': 'Fertile window approaching',
              'subtitle': 'Days 12–16 are your fertile window this cycle',
              'time': 'In ~9 days',
            },
            {
              'icon': Icons.monitor_heart_rounded,
              'color': const Color(0xFF6DBFB0),
              'title': 'Monthly health check-in',
              'subtitle': 'Time to log your symptoms & take the survey',
              'time': 'Tomorrow',
            },
          ];

          Widget notifToggle(
            String label,
            String sub,
            IconData icon,
            Color color,
            bool value,
            Function(bool) onChanged,
          ) {
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              decoration: BoxDecoration(
                color: value ? color.withOpacity(0.06) : Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color:
                      value ? color.withOpacity(0.25) : const Color(0xFFFCE7F3),
                  width: 1.5,
                ),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(9),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(icon, color: color, size: 18),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(label,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: value
                                  ? const Color(0xFF2D1B2E)
                                  : Colors.grey.shade500,
                              decoration: TextDecoration.none,
                            )),
                        const SizedBox(height: 2),
                        Text(sub,
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.grey.shade400,
                              decoration: TextDecoration.none,
                            )),
                      ],
                    ),
                  ),
                  Switch(
                    value: value,
                    onChanged: (v) {
                      onChanged(v);
                      setModalState(() {});
                    },
                    activeThumbColor: color,
                    activeTrackColor: color.withOpacity(0.25),
                    inactiveThumbColor: Colors.grey.shade300,
                    inactiveTrackColor: Colors.grey.shade100,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ],
              ),
            );
          }

          return Container(
            decoration: const BoxDecoration(
              color: Color(0xFFFFF5F8),
              borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
            ),
            padding: EdgeInsets.fromLTRB(
                20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Drag handle
                  Center(
                    child: Container(
                      margin: const EdgeInsets.only(top: 12, bottom: 16),
                      width: 44,
                      height: 5,
                      decoration: BoxDecoration(
                        color: const Color(0xFFE0C8D8),
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  ),

                  // Header row
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(12),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFFC85A7A).withOpacity(0.3),
                              blurRadius: 8,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: const Icon(Icons.notifications_rounded,
                            color: Colors.white, size: 20),
                      ),
                      const SizedBox(width: 12),
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Notifications',
                              style: TextStyle(
                                color: Color(0xFF2D1B2E),
                                fontSize: 18,
                                fontWeight: FontWeight.w800,
                                decoration: TextDecoration.none,
                              )),
                          Text('Reminders & alerts',
                              style: TextStyle(
                                color: Color(0xFFBBAACE),
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                decoration: TextDecoration.none,
                              )),
                        ],
                      ),
                      const Spacer(),
                      GestureDetector(
                        onTap: () => Navigator.pop(context),
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF5EEF5),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.close_rounded,
                              color: Color(0xFFBB8FAE), size: 18),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // ── Upcoming reminders ──────────────────────────────
                  const Text('Upcoming Reminders',
                      style: TextStyle(
                        color: Color(0xFF2D1B2E),
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        decoration: TextDecoration.none,
                      )),
                  const SizedBox(height: 12),

                  ...upcoming.map((item) => Container(
                        margin: const EdgeInsets.only(bottom: 10),
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                              color: const Color(0xFFFCE7F3), width: 1.5),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFFC85A7A).withOpacity(0.05),
                              blurRadius: 8,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(9),
                              decoration: BoxDecoration(
                                color:
                                    (item['color'] as Color).withOpacity(0.12),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Icon(item['icon'] as IconData,
                                  color: item['color'] as Color, size: 18),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(item['title'] as String,
                                      style: const TextStyle(
                                        fontSize: 13,
                                        fontWeight: FontWeight.w700,
                                        color: Color(0xFF2D1B2E),
                                        decoration: TextDecoration.none,
                                      )),
                                  const SizedBox(height: 2),
                                  Text(item['subtitle'] as String,
                                      style: TextStyle(
                                        fontSize: 11,
                                        color: Colors.grey.shade500,
                                        decoration: TextDecoration.none,
                                      )),
                                ],
                              ),
                            ),
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color:
                                    (item['color'] as Color).withOpacity(0.10),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                item['time'] as String,
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.w700,
                                  color: item['color'] as Color,
                                  decoration: TextDecoration.none,
                                ),
                              ),
                            ),
                          ],
                        ),
                      )),

                  const SizedBox(height: 24),

                  // ── Notification settings ───────────────────────────
                  const Text('Notification Settings',
                      style: TextStyle(
                        color: Color(0xFF2D1B2E),
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        decoration: TextDecoration.none,
                      )),
                  const SizedBox(height: 12),

                  notifToggle(
                      'Period Reminder',
                      'Alert 2 days before expected period',
                      Icons.water_drop_rounded,
                      const Color(0xFFC85A7A),
                      _notifPeriodReminder,
                      (v) => setState(() => _notifPeriodReminder = v)),

                  notifToggle(
                      'Fertile Window Alert',
                      'Notify when fertile days are approaching',
                      Icons.favorite_rounded,
                      const Color(0xFF9B84D4),
                      _notifFertileWindow,
                      (v) => setState(() => _notifFertileWindow = v)),

                  notifToggle(
                      'Medicine Reminder',
                      'Daily reminder to take supplements/pills',
                      Icons.medication_rounded,
                      const Color(0xFF6DBFB0),
                      _notifMedReminder,
                      (v) => setState(() => _notifMedReminder = v)),

                  notifToggle(
                      'Cycle Insights',
                      'Weekly summary of your cycle health',
                      Icons.insights_rounded,
                      const Color(0xFFE8A838),
                      _notifCycleInsights,
                      (v) => setState(() => _notifCycleInsights = v)),

                  const SizedBox(height: 8),

                  // Done button
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(18),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFFC85A7A).withOpacity(0.35),
                            blurRadius: 14,
                            offset: const Offset(0, 5),
                          ),
                        ],
                      ),
                      child: const Center(
                        child: Text(
                          'Save & Close',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.5,
                            decoration: TextDecoration.none,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFFFFF0F7),
            Color(0xFFFDE8F5),
            Color(0xFFF5E6FF),
            Color(0xFFFFFFFF)
          ],
          stops: [0.0, 0.3, 0.6, 1.0],
        ),
      ),
      child: Column(
        children: [
          _buildHeader(),
          _buildCircularCalendar(),
          _buildNavigationButtons(),
          _buildTabs(),
          _buildContent(),
        ],
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 36, 20, 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _getMonthName(selectedDate.month),
                style: const TextStyle(
                  color: Color(0xFFC85A7A),
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -0.5,
                  decoration: TextDecoration.none, // ✅
                ),
              ),
              Text(
                selectedDate.year.toString(),
                style: const TextStyle(
                  color: Color(0xFFD4A0B8),
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 1.2,
                  decoration: TextDecoration.none, // ✅
                ),
              ),
            ],
          ),
          Row(
            children: [
              Stack(
                clipBehavior: Clip.none,
                children: [
                  _buildIconButton(Icons.notifications_none_rounded,
                      onTap: _showNotificationPanel),
                  if (_hasUnread)
                    Positioned(
                      top: -2,
                      right: -2,
                      child: Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: const Color(0xFFC85A7A),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 1.5),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(width: 10),
              _buildIconButton(Icons.calendar_month_rounded,
                  onTap: _showCalendarModal, isHighlighted: true),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton(IconData icon,
      {required VoidCallback onTap, bool isHighlighted = false}) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: isHighlighted ? const Color(0xFFC85A7A) : Colors.white,
          borderRadius: BorderRadius.circular(14),
          boxShadow: [
            BoxShadow(
              color: isHighlighted
                  ? const Color(0xFFC85A7A).withOpacity(0.35)
                  : Colors.black.withOpacity(0.08),
              blurRadius: isHighlighted ? 12 : 8,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Icon(icon,
            color: isHighlighted ? Colors.white : const Color(0xFFC85A7A),
            size: 22),
      ),
    );
  }

  // ── Circular Calendar ─────────────────────────────────────────────────────

  Widget _buildCircularCalendar() {
    final int totalDays = _getDaysInMonth();
    const double radius = 138;

    return SizedBox(
      height: 320,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            width: 160,
            height: 160,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [
                  const Color(0xFFFFD6E8).withOpacity(0.6),
                  Colors.transparent
                ],
              ),
            ),
          ),
          ...List.generate(totalDays, (index) {
            final angle = (2 * math.pi / totalDays) * index - math.pi / 2;
            final x = radius * math.cos(angle);
            final y = radius * math.sin(angle);
            final day = index + 1;

            final isPeriod = periodDays.contains(day);
            final isFertile = fertileDays.contains(day);
            final isSelected = selectedDay == day;
            final isToday = day == DateTime.now().day &&
                selectedDate.month == DateTime.now().month &&
                selectedDate.year == DateTime.now().year;

            return Transform.translate(
              offset: Offset(x, y),
              child: GestureDetector(
                onTap: () {
                  setState(() {
                    selectedDay = day;
                  });
                  _updateCycleInsights(day);
                },
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: isSelected ? 42 : 36,
                  height: isSelected ? 42 : 36,
                  decoration: BoxDecoration(
                    gradient: isPeriod
                        ? const LinearGradient(
                            colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight)
                        : isFertile
                            ? const LinearGradient(
                                colors: [Color(0xFFB5A4E0), Color(0xFF9B84D4)],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight)
                            : null,
                    color: !isPeriod && !isFertile
                        ? isToday
                            ? const Color(0xFFFFD6E8)
                            : const Color(0xFFF5E6F5).withOpacity(0.6)
                        : null,
                    shape: BoxShape.circle,
                    border: isSelected
                        ? Border.all(color: const Color(0xFFC85A7A), width: 2.5)
                        : isToday && !isPeriod
                            ? Border.all(
                                color: const Color(0xFFE87DAB).withOpacity(0.5),
                                width: 1.5)
                            : null,
                    boxShadow: isPeriod
                        ? [
                            BoxShadow(
                                color:
                                    const Color(0xFFC85A7A).withOpacity(0.35),
                                blurRadius: 6,
                                offset: const Offset(0, 2))
                          ]
                        : isFertile
                            ? [
                                BoxShadow(
                                    color: const Color(0xFF9B84D4)
                                        .withOpacity(0.35),
                                    blurRadius: 6,
                                    offset: const Offset(0, 2))
                              ]
                            : null,
                  ),
                  child: Center(
                    child: Text(
                      day.toString(),
                      style: TextStyle(
                        color: isPeriod || isFertile
                            ? Colors.white
                            : isToday
                                ? const Color(0xFFC85A7A)
                                : const Color(0xFFCCA8C0),
                        fontSize: 11,
                        fontWeight: isPeriod || isFertile || isToday
                            ? FontWeight.w700
                            : FontWeight.w500,
                        decoration: TextDecoration.none, // ✅
                      ),
                    ),
                  ),
                ),
              ),
            );
          }),
          _buildCenterDisplay(),
        ],
      ),
    );
  }

  Widget _buildCenterDisplay() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          "PERIOD IN",
          style: TextStyle(
            color: Color(0xFFBB8FAE),
            fontSize: 12,
            fontWeight: FontWeight.w600,
            decoration: TextDecoration.none,
          ),
        ),
        Text(
          "$_daysUntilPeriod Days",
          style: TextStyle(
            color: Color(0xFFC85A7A),
            fontSize: 40,
            fontWeight: FontWeight.w900,
            decoration: TextDecoration.none,
          ),
        ),
        Text(
          _currentPhase,
          style: TextStyle(
            color: Color(0xFFE087A8),
            fontSize: 14,
            fontWeight: FontWeight.w600,
            decoration: TextDecoration.none,
          ),
        ),
      ],
    );
  }

  Widget _buildSymptomsLogger() {
    final symptoms = ['Cramps', 'Headache', 'Bloating', 'Acne', 'Tired'];
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "How are you feeling?",
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            children: symptoms
                .map(
                  (s) => FilterChip(
                    label: Text(s),
                    selected: _selectedSymptoms.contains(s),
                    onSelected: (val) {
                      setState(() {
                        if (val) {
                          _selectedSymptoms.add(s);
                        } else {
                          _selectedSymptoms.remove(s);
                        }
                      });
                      // Call your FastAPI /cycle/log here
                    },
                  ),
                )
                .toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildBadge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.25), width: 1),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
          decoration: TextDecoration.none, // ✅
        ),
      ),
    );
  }

  Widget _buildNavArrow(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          color: const Color(0xFFFFF0F7),
          shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFFEEC4D6), width: 1),
        ),
        child: Icon(icon, color: const Color(0xFFC85A7A), size: 20),
      ),
    );
  }

  // ── Navigation buttons + legend ───────────────────────────────────────────

  Widget _buildNavigationButtons() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _buildYearNavButton(Icons.keyboard_double_arrow_left_rounded,
              'Prev year', () => _changeYear(-1)),
          Row(
            children: [
              _buildLegendDot(const Color(0xFFC85A7A), 'Period'),
              const SizedBox(width: 12),
              _buildLegendDot(const Color(0xFF9B84D4), 'Fertile'),
            ],
          ),
          _buildYearNavButton(Icons.keyboard_double_arrow_right_rounded,
              'Next year', () => _changeYear(1)),
        ],
      ),
    );
  }

  Widget _buildYearNavButton(
      IconData icon, String tooltip, VoidCallback onTap) {
    return Tooltip(
      message: tooltip,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(10),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withOpacity(0.06),
                  blurRadius: 6,
                  offset: const Offset(0, 2))
            ],
          ),
          child: Icon(icon, color: const Color(0xFFC85A7A), size: 20),
        ),
      ),
    );
  }

  Widget _buildLegendDot(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(color: color.withOpacity(0.4), blurRadius: 4)
            ],
          ),
        ),
        const SizedBox(width: 5),
        Text(
          label,
          style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.w600,
            decoration: TextDecoration.none, // ✅
          ),
        ),
      ],
    );
  }

  // ── Tabs ──────────────────────────────────────────────────────────────────

  Widget _buildTabs() {
    return Container(
      margin: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: const Color(0xFFF5E6F5).withOpacity(0.7),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          _buildTab(0, Icons.edit_note_rounded, 'Log Day'),
          _buildTab(1, Icons.history_rounded, 'History'),
        ],
      ),
    );
  }

  Widget _buildTab(int index, IconData icon, String label) {
    final isSelected = selectedTab == index;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => selectedTab = index),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? Colors.white : Colors.transparent,
            borderRadius: BorderRadius.circular(13),
            boxShadow: isSelected
                ? [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 8,
                        offset: const Offset(0, 2))
                  ]
                : null,
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon,
                  color: isSelected
                      ? const Color(0xFFC85A7A)
                      : const Color(0xFFBBAACE),
                  size: 18),
              const SizedBox(width: 6),
              Text(
                label,
                style: TextStyle(
                  color: isSelected
                      ? const Color(0xFFC85A7A)
                      : const Color(0xFFBBAACE),
                  fontSize: 14,
                  fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                  decoration: TextDecoration.none, // ✅
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Content ───────────────────────────────────────────────────────────────

  Widget _buildContent() {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.all(20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
                color: const Color(0xFFC85A7A).withOpacity(0.06),
                blurRadius: 20,
                offset: const Offset(0, 4))
          ],
        ),
        child:
            selectedTab == 0 ? _buildLogDayContent() : _buildHistoryContent(),
      ),
    );
  }

  Widget _buildLogDayContent() {
    final day = selectedDay ?? selectedDate.day;
    final isPeriodDay = periodDays.contains(day);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    color: const Color(0xFFFFEEF5),
                    borderRadius: BorderRadius.circular(10)),
                child: const Icon(Icons.water_drop_rounded,
                    color: Color(0xFFC85A7A), size: 18),
              ),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${_getMonthName(selectedDate.month)} $day',
                    style: const TextStyle(
                      color: Color(0xFF2D1B2E),
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      decoration: TextDecoration.none, // ✅
                    ),
                  ),
                  Text(
                    isPeriodDay ? 'Period day tracked ✓' : 'No period logged',
                    style: TextStyle(
                      color: isPeriodDay
                          ? const Color(0xFFC85A7A)
                          : const Color(0xFFBBAABB),
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      decoration: TextDecoration.none, // ✅
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 20),
          GestureDetector(
            onTap: () {
              setState(() {
                if (isPeriodDay) {
                  periodDays.remove(day);
                } else {
                  periodDays.add(day);
                }
              });
              _savePeriodData();
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: isPeriodDay
                    ? const LinearGradient(
                        colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight)
                    : const LinearGradient(
                        colors: [Color(0xFFFFF0F7), Color(0xFFFDE8F0)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight),
                borderRadius: BorderRadius.circular(18),
                border: isPeriodDay
                    ? null
                    : Border.all(color: const Color(0xFFEEC4D6), width: 1.5),
                boxShadow: isPeriodDay
                    ? [
                        BoxShadow(
                            color: const Color(0xFFC85A7A).withOpacity(0.3),
                            blurRadius: 12,
                            offset: const Offset(0, 4))
                      ]
                    : null,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    isPeriodDay
                        ? Icons.check_circle_rounded
                        : Icons.water_drop_outlined,
                    color: isPeriodDay ? Colors.white : const Color(0xFFC85A7A),
                    size: 22,
                  ),
                  const SizedBox(width: 10),
                  Text(
                    isPeriodDay ? 'Period tracked!' : 'Mark as period day',
                    style: TextStyle(
                      color:
                          isPeriodDay ? Colors.white : const Color(0xFFC85A7A),
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      decoration: TextDecoration.none, // ✅
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: const Color(0xFFF9F0FD),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFFE5D4F0), width: 1),
            ),
            child: Row(
              children: [
                const Icon(Icons.lightbulb_outline_rounded,
                    color: Color(0xFF9B84D4), size: 18),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Tap any day on the circle to select it, then mark it here',
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: 12,
                      height: 1.4,
                      decoration: TextDecoration.none, // ✅
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                  child: _buildStatCard(
                      '${periodDays.length}',
                      'Days this month',
                      const Color(0xFFC85A7A),
                      Icons.calendar_today_rounded)),
              const SizedBox(width: 12),
              Expanded(
                  child: _buildStatCard('${fertileDays.length}', 'Fertile days',
                      const Color(0xFF9B84D4), Icons.favorite_rounded)),
            ],
          ),
          const SizedBox(height: 16),
          _buildSymptomsLogger(),
        ],
      ),
    );
  }

  Widget _buildStatCard(
      String value, String label, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.2), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 22,
              fontWeight: FontWeight.w800,
              decoration: TextDecoration.none, // ✅
            ),
          ),
          Text(
            label,
            style: TextStyle(
              color: color.withOpacity(0.7),
              fontSize: 11,
              fontWeight: FontWeight.w500,
              decoration: TextDecoration.none, // ✅
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryContent() {
    final history = [
      {
        'month': 'January 2026',
        'days': 'Days: 1, 2, 3, 4, 5',
        'duration': '5 days',
        'cycle': '28d cycle'
      },
      {
        'month': 'December 2025',
        'days': 'Days: 3, 4, 5, 6',
        'duration': '4 days',
        'cycle': '27d cycle'
      },
      {
        'month': 'November 2025',
        'days': 'Days: 5, 6, 7, 8, 9',
        'duration': '5 days',
        'cycle': '29d cycle'
      },
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Period History',
              style: TextStyle(
                color: Color(0xFF2D1B2E),
                fontSize: 16,
                fontWeight: FontWeight.w700,
                decoration: TextDecoration.none, // ✅
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                  color: const Color(0xFFFFEEF5),
                  borderRadius: BorderRadius.circular(20)),
              child: const Text(
                '3 months',
                style: TextStyle(
                  color: Color(0xFFC85A7A),
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  decoration: TextDecoration.none, // ✅
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        Expanded(
          child: ListView.separated(
            itemCount: history.length,
            separatorBuilder: (_, __) => const SizedBox(height: 10),
            itemBuilder: (context, index) => _buildHistoryItem(history[index]),
          ),
        ),
      ],
    );
  }

  Widget _buildHistoryItem(Map<String, String> item) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF5F9),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFEED4E0), width: 1),
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                  colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight),
              borderRadius: BorderRadius.circular(14),
              boxShadow: [
                BoxShadow(
                    color: const Color(0xFFC85A7A).withOpacity(0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 3))
              ],
            ),
            child: const Icon(Icons.water_drop_rounded,
                color: Colors.white, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item['month']!,
                  style: const TextStyle(
                    color: Color(0xFF2D1B2E),
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.none, // ✅
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  item['days']!,
                  style: TextStyle(
                    color: Colors.grey[500],
                    fontSize: 12,
                    decoration: TextDecoration.none, // ✅
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                    color: const Color(0xFFC85A7A),
                    borderRadius: BorderRadius.circular(8)),
                child: Text(
                  item['duration']!,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.none, // ✅
                  ),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                item['cycle']!,
                style: const TextStyle(
                  color: Color(0xFF9B84D4),
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                  decoration: TextDecoration.none, // ✅
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─── Full Calendar Modal ──────────────────────────────────────────────────────

class _FullCalendarModal extends StatefulWidget {
  final DateTime selectedDate;
  final List<int> periodDays;
  final List<int> fertileDays;
  final Function(int) onDayTap;
  final Function(DateTime) onMonthChanged;
  final ScrollController scrollController;

  const _FullCalendarModal({
    required this.selectedDate,
    required this.periodDays,
    required this.fertileDays,
    required this.onDayTap,
    required this.onMonthChanged,
    required this.scrollController,
  });

  @override
  State<_FullCalendarModal> createState() => _FullCalendarModalState();
}

class _FullCalendarModalState extends State<_FullCalendarModal> {
  late DateTime _currentDate;
  late List<int> _periodDays;
  late List<int> _fertileDays;
  int? _selectedDay;

  @override
  void initState() {
    super.initState();
    _currentDate = widget.selectedDate;
    _periodDays = List.from(widget.periodDays);
    _fertileDays = List.from(widget.fertileDays);
  }

  int get _daysInMonth =>
      DateTime(_currentDate.year, _currentDate.month + 1, 0).day;
  int get _firstWeekday =>
      DateTime(_currentDate.year, _currentDate.month, 1).weekday;

  String _getMonthName(int month) {
    const months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December'
    ];
    return months[month - 1];
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: ListView(
        controller: widget.scrollController,
        padding: EdgeInsets.zero,
        physics: const ClampingScrollPhysics(),
        children: [
          // Drag handle
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 12, bottom: 4),
              width: 44,
              height: 5,
              decoration: BoxDecoration(
                  color: const Color(0xFFE0C8D8),
                  borderRadius: BorderRadius.circular(3)),
            ),
          ),

          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 4),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                        colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight),
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                          color: const Color(0xFFC85A7A).withOpacity(0.3),
                          blurRadius: 8,
                          offset: const Offset(0, 3))
                    ],
                  ),
                  child: const Icon(Icons.calendar_month_rounded,
                      color: Colors.white, size: 20),
                ),
                const SizedBox(width: 12),
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Period Calendar',
                      style: TextStyle(
                        color: Color(0xFF2D1B2E),
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        decoration: TextDecoration.none, // ✅
                      ),
                    ),
                    Text(
                      'Tap a day to toggle period tracking',
                      style: TextStyle(
                        color: Color(0xFFBBAACE),
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        decoration: TextDecoration.none, // ✅
                      ),
                    ),
                  ],
                ),
                const Spacer(),
                GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                        color: const Color(0xFFF5EEF5),
                        borderRadius: BorderRadius.circular(10)),
                    child: const Icon(Icons.close_rounded,
                        color: Color(0xFFBB8FAE), size: 18),
                  ),
                ),
              ],
            ),
          ),

          // Month navigation
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _navButton(Icons.chevron_left_rounded, () {
                  setState(() {
                    _currentDate =
                        DateTime(_currentDate.year, _currentDate.month - 1, 1);
                    _selectedDay = null;
                  });
                }),
                Column(
                  children: [
                    Text(
                      _getMonthName(_currentDate.month),
                      style: const TextStyle(
                        color: Color(0xFF2D1B2E),
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.5,
                        decoration: TextDecoration.none, // ✅
                      ),
                    ),
                    Text(
                      _currentDate.year.toString(),
                      style: const TextStyle(
                        color: Color(0xFFBBAACE),
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.5,
                        decoration: TextDecoration.none, // ✅
                      ),
                    ),
                  ],
                ),
                _navButton(Icons.chevron_right_rounded, () {
                  setState(() {
                    _currentDate =
                        DateTime(_currentDate.year, _currentDate.month + 1, 1);
                    _selectedDay = null;
                  });
                }),
              ],
            ),
          ),

          // Weekday header row
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                  .map((d) => Expanded(
                        child: Center(
                          child: Text(
                            d,
                            style: TextStyle(
                              color: (d == 'Sat' || d == 'Sun')
                                  ? const Color(0xFFE087A8)
                                  : const Color(0xFFBBAACC),
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.3,
                              decoration: TextDecoration.none, // ✅
                            ),
                          ),
                        ),
                      ))
                  .toList(),
            ),
          ),

          const SizedBox(height: 8),

          // Calendar grid
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: _buildCalendarTable(),
          ),

          const SizedBox(height: 16),

          // Legend
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF5F9),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFEED4E0), width: 1),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _legendItem(const Color(0xFFC85A7A), 'Period'),
                  _legendItem(const Color(0xFF9B84D4), 'Fertile'),
                  _legendItem(const Color(0xFFFF9EC5), 'Today'),
                  _legendItem(const Color(0xFFE8D8F0), 'Selected'),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Done button
          Padding(
            padding: EdgeInsets.fromLTRB(
                20, 0, 20, MediaQuery.of(context).padding.bottom + 20),
            child: GestureDetector(
              onTap: () {
                widget.onMonthChanged(_currentDate);
                Navigator.pop(context);
              },
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                      colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight),
                  borderRadius: BorderRadius.circular(18),
                  boxShadow: [
                    BoxShadow(
                        color: const Color(0xFFC85A7A).withOpacity(0.35),
                        blurRadius: 14,
                        offset: const Offset(0, 5))
                  ],
                ),
                child: const Center(
                  child: Text(
                    'Done',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.5,
                      decoration: TextDecoration.none, // ✅
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCalendarTable() {
    final int startOffset = _firstWeekday - 1;
    final int totalDays = _daysInMonth;
    final List<int?> cells = [
      ...List<int?>.filled(startOffset, null),
      ...List<int?>.generate(totalDays, (i) => i + 1),
    ];
    while (cells.length % 7 != 0) {
      cells.add(null);
    }
    final int rowCount = cells.length ~/ 7;

    return Column(
      children: List.generate(rowCount, (rowIndex) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            children: List.generate(7, (colIndex) {
              final int? day = cells[rowIndex * 7 + colIndex];
              if (day == null) return const Expanded(child: SizedBox());
              return Expanded(child: _buildDayCell(day, colIndex));
            }),
          ),
        );
      }),
    );
  }

  Widget _buildDayCell(int day, int colIndex) {
    final isPeriod = _periodDays.contains(day);
    final isFertile = _fertileDays.contains(day);
    final isSelected = _selectedDay == day;
    final isWeekend = colIndex >= 5;
    final isToday = day == DateTime.now().day &&
        _currentDate.month == DateTime.now().month &&
        _currentDate.year == DateTime.now().year;

    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedDay = day;
          if (_periodDays.contains(day)) {
            _periodDays.remove(day);
          } else {
            _periodDays.add(day);
          }
        });
        widget.onDayTap(day);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        margin: const EdgeInsets.symmetric(horizontal: 2),
        height: 44,
        decoration: BoxDecoration(
          gradient: isPeriod
              ? const LinearGradient(
                  colors: [Color(0xFFE87DAB), Color(0xFFC85A7A)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight)
              : isFertile
                  ? const LinearGradient(
                      colors: [Color(0xFFB5A4E0), Color(0xFF9B84D4)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight)
                  : null,
          color: (!isPeriod && !isFertile)
              ? isToday
                  ? const Color(0xFFFFD6E8)
                  : isSelected
                      ? const Color(0xFFEED8F0)
                      : Colors.transparent
              : null,
          borderRadius: BorderRadius.circular(12),
          border: isSelected && !isPeriod
              ? Border.all(color: const Color(0xFFC85A7A), width: 2)
              : isToday && !isPeriod
                  ? Border.all(
                      color: const Color(0xFFE87DAB).withOpacity(0.5),
                      width: 1.5)
                  : null,
          boxShadow: isPeriod
              ? [
                  BoxShadow(
                      color: const Color(0xFFC85A7A).withOpacity(0.28),
                      blurRadius: 6,
                      offset: const Offset(0, 3))
                ]
              : isFertile
                  ? [
                      BoxShadow(
                          color: const Color(0xFF9B84D4).withOpacity(0.28),
                          blurRadius: 6,
                          offset: const Offset(0, 3))
                    ]
                  : null,
        ),
        child: Center(
          child: Text(
            day.toString(),
            style: TextStyle(
              color: isPeriod || isFertile
                  ? Colors.white
                  : isToday
                      ? const Color(0xFFC85A7A)
                      : isWeekend
                          ? const Color(0xFFE087A8)
                          : const Color(0xFF2D1B2E),
              fontSize: 14,
              fontWeight: isPeriod || isFertile || isToday || isSelected
                  ? FontWeight.w700
                  : FontWeight.w500,
              decoration: TextDecoration.none, // ✅
            ),
          ),
        ),
      ),
    );
  }

  Widget _navButton(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: const Color(0xFFFFF0F7),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFEEC4D6), width: 1),
        ),
        child: Icon(icon, color: const Color(0xFFC85A7A), size: 22),
      ),
    );
  }

  Widget _legendItem(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(4),
            boxShadow: [
              BoxShadow(color: color.withOpacity(0.4), blurRadius: 4)
            ],
          ),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(
            color: color,
            fontSize: 12,
            fontWeight: FontWeight.w600,
            decoration: TextDecoration.none, // ✅
          ),
        ),
      ],
    );
  }
}
