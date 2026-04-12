// ═══════════════════════════════════════════════════════════════════════════
// cycle.dart  –  She Health · Advanced Period Cycle Monitoring
// Flutter + FastAPI + MongoDB
// ═══════════════════════════════════════════════════════════════════════════
//
// NEW FEATURES OVER v1
// ─────────────────────
// ✅ Log ANY past cycle (history back-filling with date picker)
// ✅ Daily symptom logger — cramps, headache, bloating, backache, nausea,
//    breast tenderness, acne, spotting
// ✅ Mood tracker — happy, calm, anxious, irritable, sad, energetic, tired
// ✅ Flow intensity — none / spotting / light / medium / heavy
// ✅ Multi-cycle calendar — scrolls months, shows ALL logged cycles coloured
// ✅ Cycle regularity badge — regular / slightly irregular / irregular
// ✅ Average cycle trend chart (bar chart — pure Flutter, no extra package)
// ✅ Delete a history entry (swipe or long-press)
// ✅ Edit an existing cycle entry
// ✅ Symptom summary card per cycle
// ✅ Prediction confidence indicator
// ✅ Full offline-first with local cache fallback
// ═══════════════════════════════════════════════════════════════════════════

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// ── Config ───────────────────────────────────────────────────────────────────
const String kApiBase = 'http://10.0.2.2:8000';
const String kQAnswersKey = 'questionnaire_answers';
const String kPeriodDataPrefsKey = 'period_days_v1';
const String kCycleHistoryKey = 'cycle_history_v2';
const String kDailyLogsKey = 'daily_logs_v1';

// ── Colours ───────────────────────────────────────────────────────────────────
const _pink = Color(0xFFC85A7A);
const _pinkLight = Color(0xFFE87DAB);
const _purple = Color(0xFF9B84D4);
const _teal = Color(0xFF6DBFB0);
const _amber = Color(0xFFE8A838);
const _bg = Color(0xFFFFF0F7);
const _textDark = Color(0xFF2D1B2E);
const _textMid = Color(0xFFBBAACC);

// ═══════════════════════════════════════════════════════════════════════════
// MODELS
// ═══════════════════════════════════════════════════════════════════════════

enum FlowIntensity { none, spotting, light, medium, heavy }

extension FlowIntensityExt on FlowIntensity {
  String get label => ['None', 'Spotting', 'Light', 'Medium', 'Heavy'][index];
  Color get color => [
        Colors.transparent,
        const Color(0xFFFFD6E8),
        const Color(0xFFFFAACC),
        _pink,
        const Color(0xFF8B1A4A)
      ][index];
  String get emoji => ['—', '💧', '🩸', '🩸🩸', '🩸🩸🩸'][index];
}

enum Mood { happy, calm, anxious, irritable, sad, energetic, tired }

extension MoodExt on Mood {
  String get label => [
        'Happy',
        'Calm',
        'Anxious',
        'Irritable',
        'Sad',
        'Energetic',
        'Tired'
      ][index];
  String get emoji => ['😊', '😌', '😟', '😤', '😢', '⚡', '😴'][index];
  Color get color => [
        const Color(0xFFFFD700),
        _teal,
        _amber,
        _pink,
        _purple,
        const Color(0xFF4CAF50),
        _textMid
      ][index];
}

const kSymptoms = [
  'Cramps',
  'Headache',
  'Bloating',
  'Backache',
  'Nausea',
  'Breast Tenderness',
  'Acne',
  'Spotting',
  'Fatigue',
  'Mood Swings',
  'Hot Flashes',
  'Insomnia',
];

const kSymptomEmojis = [
  '🤕',
  '🤯',
  '🫃',
  '🔙',
  '🤢',
  '🤱',
  '😣',
  '💧',
  '😴',
  '🎭',
  '🥵',
  '😶',
];

// ── DailyLog ─────────────────────────────────────────────────────────────────
class DailyLog {
  final String date; // yyyy-MM-dd
  final FlowIntensity flow;
  final List<Mood> moods;
  final List<String> symptoms;
  final String note;

  DailyLog({
    required this.date,
    this.flow = FlowIntensity.none,
    this.moods = const [],
    this.symptoms = const [],
    this.note = '',
  });

  Map<String, dynamic> toJson() => {
        'date': date,
        'flow': flow.index,
        'moods': moods.map((m) => m.index).toList(),
        'symptoms': symptoms,
        'note': note,
      };

  factory DailyLog.fromJson(Map<String, dynamic> j) => DailyLog(
        date: j['date'] as String,
        flow: FlowIntensity.values[(j['flow'] as int?) ?? 0],
        moods: ((j['moods'] as List?) ?? [])
            .map((e) => Mood.values[e as int])
            .toList(),
        symptoms: ((j['symptoms'] as List?) ?? []).cast<String>(),
        note: (j['note'] as String?) ?? '',
      );
}

// ── CycleData ─────────────────────────────────────────────────────────────────
class CycleData {
  final String? id;
  final DateTime cycleStartDate;
  final int cycleLength;
  final int periodDuration;
  final bool isHistorical; // true = back-filled by user

  const CycleData({
    this.id,
    required this.cycleStartDate,
    required this.cycleLength,
    required this.periodDuration,
    this.isHistorical = false,
  });

  List<DateTime> get periodDays => List.generate(
      periodDuration, (i) => cycleStartDate.add(Duration(days: i)));

  DateTime get ovulationDay =>
      cycleStartDate.add(Duration(days: cycleLength - 14));

  List<DateTime> get fertileDays =>
      List.generate(6, (i) => ovulationDay.subtract(Duration(days: 5 - i)));

  DateTime get nextPeriodStart =>
      cycleStartDate.add(Duration(days: cycleLength));

  List<DateTime> get pmsDays =>
      List.generate(5, (i) => nextPeriodStart.subtract(Duration(days: 5 - i)));

  int get currentCycleDay {
    final d = DateTime.now().difference(cycleStartDate).inDays + 1;
    return d.clamp(1, cycleLength);
  }

  String get currentPhase {
    final day = currentCycleDay;
    if (day <= periodDuration) return 'Menstrual Phase';
    if (day <= cycleLength - 14 - 5) return 'Follicular Phase';
    if (day <= cycleLength - 14 + 1) return 'Ovulatory Phase';
    return 'Luteal Phase';
  }

  String get phaseEmoji {
    switch (currentPhase) {
      case 'Menstrual Phase':
        return '🩸';
      case 'Follicular Phase':
        return '🌱';
      case 'Ovulatory Phase':
        return '🌸';
      default:
        return '🌙';
    }
  }

  int get daysUntilNext => nextPeriodStart.difference(DateTime.now()).inDays;

  Map<String, dynamic> toJson() => {
        if (id != null) 'id': id,
        'cycle_start_date': cycleStartDate.toIso8601String(),
        'cycle_length': cycleLength,
        'period_duration': periodDuration,
        'is_historical': isHistorical,
      };

  factory CycleData.fromJson(Map<String, dynamic> j) => CycleData(
        id: j['id'] as String?,
        cycleStartDate: DateTime.parse(j['cycle_start_date'] as String),
        cycleLength: (j['cycle_length'] as num).toInt(),
        periodDuration: (j['period_duration'] as num).toInt(),
        isHistorical: (j['is_historical'] as bool?) ?? false,
      );

  CycleData copyWith({
    String? id,
    DateTime? cycleStartDate,
    int? cycleLength,
    int? periodDuration,
    bool? isHistorical,
  }) =>
      CycleData(
        id: id ?? this.id,
        cycleStartDate: cycleStartDate ?? this.cycleStartDate,
        cycleLength: cycleLength ?? this.cycleLength,
        periodDuration: periodDuration ?? this.periodDuration,
        isHistorical: isHistorical ?? this.isHistorical,
      );
}

// ── RegularityStatus ──────────────────────────────────────────────────────────
enum Regularity { regular, slightlyIrregular, irregular }

extension RegularityExt on Regularity {
  String get label => ['Regular', 'Slightly Irregular', 'Irregular'][index];
  Color get color => [_teal, _amber, _pink][index];
  IconData get icon => [
        Icons.check_circle_rounded,
        Icons.info_rounded,
        Icons.warning_rounded,
      ][index];
}

Regularity calcRegularity(List<CycleData> history) {
  if (history.length < 2) return Regularity.regular;
  final lens = history.map((e) => e.cycleLength).toList();
  final diff = lens.reduce(math.max) - lens.reduce(math.min);
  if (diff <= 3) return Regularity.regular;
  if (diff <= 7) return Regularity.slightlyIrregular;
  return Regularity.irregular;
}

// ═══════════════════════════════════════════════════════════════════════════
// API SERVICE
// ═══════════════════════════════════════════════════════════════════════════

class CycleApi {
  static Future<bool> logCycle(CycleData cd, String userId) async {
    try {
      final r = await http
          .post(
            Uri.parse('$kApiBase/api/cycle/log'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'user_id': userId, ...cd.toJson()}),
          )
          .timeout(const Duration(seconds: 10));
      return r.statusCode == 200 || r.statusCode == 201;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> updateCycle(CycleData cd, String userId) async {
    if (cd.id == null) return false;
    try {
      final r = await http
          .put(
            Uri.parse('$kApiBase/api/cycle/${cd.id}'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'user_id': userId, ...cd.toJson()}),
          )
          .timeout(const Duration(seconds: 10));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> deleteCycle(String cycleId, String userId) async {
    try {
      final r = await http
          .delete(
            Uri.parse('$kApiBase/api/cycle/$cycleId?user_id=$userId'),
          )
          .timeout(const Duration(seconds: 10));
      return r.statusCode == 204 || r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<List<CycleData>> fetchHistory(String userId) async {
    try {
      final r = await http
          .get(
            Uri.parse('$kApiBase/api/cycle/history?user_id=$userId&limit=24'),
          )
          .timeout(const Duration(seconds: 10));
      if (r.statusCode != 200) return [];
      final List data = jsonDecode(r.body) as List;
      return data
          .map((e) => CycleData.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<bool> logDailyLog(DailyLog log, String userId) async {
    try {
      final r = await http
          .post(
            Uri.parse('$kApiBase/api/cycle/daily-log'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'user_id': userId, ...log.toJson()}),
          )
          .timeout(const Duration(seconds: 10));
      return r.statusCode == 200 || r.statusCode == 201;
    } catch (_) {
      return false;
    }
  }

  static Future<List<DailyLog>> fetchDailyLogs(String userId) async {
    try {
      final r = await http
          .get(
            Uri.parse('$kApiBase/api/cycle/daily-logs?user_id=$userId'),
          )
          .timeout(const Duration(seconds: 10));
      if (r.statusCode != 200) return [];
      final List data = jsonDecode(r.body) as List;
      return data
          .map((e) => DailyLog.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// LOCAL CACHE
// ═══════════════════════════════════════════════════════════════════════════

class LocalCache {
  static Future<List<CycleData>> loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(kCycleHistoryKey);
      if (raw == null) return [];
      final List data = jsonDecode(raw) as List;
      return data
          .map((e) => CycleData.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveHistory(List<CycleData> history) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        kCycleHistoryKey, jsonEncode(history.map((e) => e.toJson()).toList()));
  }

  static Future<Map<String, DailyLog>> loadDailyLogs() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(kDailyLogsKey);
      if (raw == null) return {};
      final Map<String, dynamic> data = jsonDecode(raw) as Map<String, dynamic>;
      return data.map(
          (k, v) => MapEntry(k, DailyLog.fromJson(v as Map<String, dynamic>)));
    } catch (_) {
      return {};
    }
  }

  static Future<void> saveDailyLogs(Map<String, DailyLog> logs) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        kDailyLogsKey, jsonEncode(logs.map((k, v) => MapEntry(k, v.toJson()))));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WIDGET
// ═══════════════════════════════════════════════════════════════════════════

class PeriodCalendarWidget extends StatefulWidget {
  const PeriodCalendarWidget({Key? key}) : super(key: key);
  @override
  State<PeriodCalendarWidget> createState() => _PeriodCalendarWidgetState();
}

class _PeriodCalendarWidgetState extends State<PeriodCalendarWidget>
    with TickerProviderStateMixin {
  List<CycleData> _history = [];
  Map<String, DailyLog> _dailyLogs = {};
  bool _loading = true;
  bool _saving = false;
  String _userId = 'demo_user';
  int _tab = 0;
  DateTime _calMonth = DateTime(DateTime.now().year, DateTime.now().month);
  bool _hasUnread = true;

  // Notification toggles
  bool _nPeriod = true, _nFertile = true, _nMed = false, _nInsights = true;

  late AnimationController _pulse;
  late Animation<double> _pulseAnim;
  late AnimationController _fade;
  late Animation<double> _fadeAnim;

  // Helpers
  CycleData? get _latest {
    if (_history.isEmpty) return null;
    final sorted = [..._history]
      ..sort((a, b) => b.cycleStartDate.compareTo(a.cycleStartDate));
    return sorted.first;
  }

  Regularity get _regularity => calcRegularity(_history);

  double get _predictionConfidence {
    if (_history.length >= 6) return 0.95;
    if (_history.length >= 3) return 0.80;
    if (_history.length >= 1) return 0.60;
    return 0.0;
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _pulse =
        AnimationController(vsync: this, duration: const Duration(seconds: 2))
          ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.95, end: 1.05)
        .animate(CurvedAnimation(parent: _pulse, curve: Curves.easeInOut));
    _fade = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));
    _fadeAnim = CurvedAnimation(parent: _fade, curve: Curves.easeOut);
    _init();
  }

  Future<void> _init() async {
    final prefs = await SharedPreferences.getInstance();
    _userId = prefs.getString('user_id') ?? 'demo_user';

    // Load local cache first (instant UI)
    final local = await LocalCache.loadHistory();
    final logs = await LocalCache.loadDailyLogs();
    if (mounted)
      setState(() {
        _history = local;
        _dailyLogs = logs;
        _loading = false;
      });
    _fade.forward();

    // Then sync with server
    final remote = await CycleApi.fetchHistory(_userId);
    if (remote.isNotEmpty && mounted) {
      await LocalCache.saveHistory(remote);
      setState(() => _history = remote);
    }
    final remoteLogs = await CycleApi.fetchDailyLogs(_userId);
    if (remoteLogs.isNotEmpty && mounted) {
      final map = {for (var l in remoteLogs) l.date: l};
      await LocalCache.saveDailyLogs(map);
      setState(() => _dailyLogs = map);
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    _fade.dispose();
    super.dispose();
  }

  // ── Questionnaire helpers ─────────────────────────────────────────────────

  Future<({int cycleLen, int periodDur})> _qaValues() async {
    final prefs = await SharedPreferences.getInstance();
    int cl = 28, pd = 5;
    try {
      final raw = prefs.getString(kQAnswersKey);
      if (raw != null) {
        final m = jsonDecode(raw) as Map<String, dynamic>;
        cl = int.tryParse(m['Q_CYCLE_LENGTH'].toString()) ?? 28;
        pd = int.tryParse(m['Q_PERIOD_DURATION'].toString()) ?? 5;
        cl = cl.clamp(21, 45);
        pd = pd.clamp(1, 10);
      }
    } catch (_) {}
    // Average recent cycles for better accuracy
    if (_history.length >= 2) {
      final recent = _history.take(3).map((e) => e.cycleLength).toList();
      cl = (recent.reduce((a, b) => a + b) / recent.length).round();
    }
    return (cycleLen: cl, periodDur: pd);
  }

  // ── Save / Delete cycle ───────────────────────────────────────────────────

  Future<void> _saveCycle(CycleData cd, {bool isEdit = false}) async {
    setState(() => _saving = true);

    // Update history list
    if (isEdit) {
      _history = _history
          .map((e) => e.cycleStartDate == cd.cycleStartDate ? cd : e)
          .toList();
    } else {
      // Remove duplicate same start date
      _history
          .removeWhere((e) => _sameDay(e.cycleStartDate, cd.cycleStartDate));
      _history.add(cd);
    }
    _history.sort((a, b) => b.cycleStartDate.compareTo(a.cycleStartDate));
    await LocalCache.saveHistory(_history);

    final ok = isEdit
        ? await CycleApi.updateCycle(cd, _userId)
        : await CycleApi.logCycle(cd, _userId);

    if (mounted) {
      setState(() => _saving = false);
      _snack(ok ? '✅ Cycle saved!' : '💾 Saved locally (offline)');
    }
  }

  Future<void> _deleteCycle(CycleData cd) async {
    final confirm = await _confirmDialog(
        'Delete Cycle', 'Remove cycle starting ${_fmt(cd.cycleStartDate)}?');
    if (!confirm) return;

    setState(() {
      _history
          .removeWhere((e) => _sameDay(e.cycleStartDate, cd.cycleStartDate));
    });
    await LocalCache.saveHistory(_history);
    if (cd.id != null) await CycleApi.deleteCycle(cd.id!, _userId);
    _snack('🗑 Cycle removed');
  }

  Future<bool> _confirmDialog(String title, String msg) async {
    return await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            title: Text(title,
                style: const TextStyle(
                    color: _textDark, fontWeight: FontWeight.w800)),
            content: Text(msg, style: const TextStyle(color: _textMid)),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(ctx, false),
                  child:
                      const Text('Cancel', style: TextStyle(color: _textMid))),
              TextButton(
                  onPressed: () => Navigator.pop(ctx, true),
                  child: const Text('Delete',
                      style: TextStyle(
                          color: _pink, fontWeight: FontWeight.w700))),
            ],
          ),
        ) ??
        false;
  }

  // ── Save daily log ─────────────────────────────────────────────────────────

  Future<void> _saveDailyLog(DailyLog log) async {
    _dailyLogs[log.date] = log;
    await LocalCache.saveDailyLogs(_dailyLogs);
    await CycleApi.logDailyLog(log, _userId);
    if (mounted) setState(() {});
    _snack('✅ Daily log saved!');
  }

  // ── Build ──────────────────────────────────────────────────────────────────

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
            Colors.white
          ],
          stops: [0, .3, .6, 1],
        ),
      ),
      child: _loading
          ? const _Loader()
          : FadeTransition(
              opacity: _fadeAnim,
              child: Column(children: [
                _header(),
                Expanded(
                    child: SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  child: Column(children: [
                    _ring(),
                    const SizedBox(height: 12),
                    if (_latest != null) _phaseCard(),
                    const SizedBox(height: 12),
                    _tabs(),
                    _tabContent(),
                    const SizedBox(height: 32),
                  ]),
                )),
              ]),
            ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _header() {
    final now = DateTime.now();
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
                  letterSpacing: -.5,
                  decoration: TextDecoration.none)),
              Text('${now.year}',
                  style: const TextStyle(
                      color: Color(0xFFD4A0B8),
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 1.2,
                      decoration: TextDecoration.none)),
            ]),
          _iconBtn(Icons.add_rounded,
              highlighted: true, onTap: () => _showLogSheet()),
          const SizedBox(width: 8),
          _iconBtn(Icons.history_rounded,
              onTap: () => _showLogSheet(isHistory: true)),
          const SizedBox(width: 8),
          Stack(clipBehavior: Clip.none, children: [
            _iconBtn(Icons.notifications_none_rounded, onTap: _showNotifPanel),
            if (_hasUnread)
              Positioned(
                  top: -2,
                  right: -2,
                  child: Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                          color: _pink,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 1.5)))),
          ]),
        ],
      ),
    );
  }

  Widget _iconBtn(IconData icon,
      {bool highlighted = false, VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: highlighted ? _pink : Colors.white,
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              BoxShadow(
                  color: highlighted
                      ? _pink.withOpacity(.35)
                      : Colors.black.withOpacity(.08),
                  blurRadius: highlighted ? 12 : 8,
                  offset: const Offset(0, 3))
            ],
          ),
          child:
              Icon(icon, color: highlighted ? Colors.white : _pink, size: 22)),
    );
  }

  // ── Radial ring ────────────────────────────────────────────────────────────

  Widget _ring() {
    if (_latest == null) return _emptyRing();
    final cd = _latest!;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      child: Column(children: [
        Center(
            child: ScaleTransition(
          scale: _pulseAnim,
          child: SizedBox(
            width: 220,
            height: 220,
            child: Stack(alignment: Alignment.center, children: [
              CustomPaint(
                  size: const Size(220, 220),
                  painter: _RingPainter(
                    progress: cd.currentCycleDay / cd.cycleLength,
                    periodDuration: cd.periodDuration,
                    cycleLength: cd.cycleLength,
                  )),
              Column(mainAxisSize: MainAxisSize.min, children: [
                Text(cd.phaseEmoji,
                    style: const TextStyle(
                        fontSize: 30, decoration: TextDecoration.none)),
                const SizedBox(height: 4),
                Text('Day ${cd.currentCycleDay}',
                    style: const TextStyle(
                        color: _textDark,
                        fontSize: 26,
                        fontWeight: FontWeight.w900,
                        decoration: TextDecoration.none)),
                Text('of ${cd.cycleLength}',
                    style: const TextStyle(
                        color: _textMid,
                        fontSize: 13,
                        decoration: TextDecoration.none)),
                const SizedBox(height: 6),
                _pillChip(
                  cd.daysUntilNext > 0
                      ? 'Next in ${cd.daysUntilNext}d'
                      : cd.daysUntilNext == 0
                          ? 'Period due today!'
                          : 'Overdue ${-cd.daysUntilNext}d',
                  _pink,
                ),
              ]),
            ]),
          ),
        )),
        const SizedBox(height: 12),
        // Regularity + confidence badges
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          _badge(_regularity.icon, _regularity.label, _regularity.color),
          const SizedBox(width: 10),
          _badge(Icons.analytics_rounded,
              '${(_predictionConfidence * 100).toInt()}% confidence', _purple),
          if (_saving) ...[
            const SizedBox(width: 10),
            const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2, color: _pink))
          ],
        ]),
      ]),
    );
  }

  Widget _emptyRing() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        child: Center(
            child: Container(
          width: 220,
          height: 220,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.white,
            boxShadow: [
              BoxShadow(
                  color: _pink.withOpacity(.12),
                  blurRadius: 24,
                  offset: const Offset(0, 8))
            ],
          ),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            const Text('🩷',
                style:
                    TextStyle(fontSize: 40, decoration: TextDecoration.none)),
            const SizedBox(height: 8),
            const Text('Log your cycle',
                style: TextStyle(
                    color: _textDark,
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    decoration: TextDecoration.none)),
            const SizedBox(height: 4),
            const Text('to see predictions',
                style: TextStyle(
                    color: _textMid,
                    fontSize: 12,
                    decoration: TextDecoration.none)),
            const SizedBox(height: 14),
            GestureDetector(
                onTap: () => _showLogSheet(),
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                          color: _pink.withOpacity(.35),
                          blurRadius: 12,
                          offset: const Offset(0, 4))
                    ],
                  ),
                  child: const Text('+ Log Cycle',
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          decoration: TextDecoration.none)),
                )),
          ]),
        )),
      );

  Widget _badge(IconData icon, String label, Color color) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: color.withOpacity(.10),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(.25)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 5),
          Text(label,
              style: TextStyle(
                  color: color,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  decoration: TextDecoration.none)),
        ]),
      );

  // ── Phase card ─────────────────────────────────────────────────────────────

  Widget _phaseCard() {
    final cd = _latest!;
    final color = _phaseColor(cd.currentPhase);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(.2), width: 1.5),
          boxShadow: [
            BoxShadow(
                color: color.withOpacity(.07),
                blurRadius: 16,
                offset: const Offset(0, 5))
          ],
        ),
        child: Column(children: [
          Row(children: [
            Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                    color: color.withOpacity(.12),
                    borderRadius: BorderRadius.circular(12)),
                child: Icon(Icons.self_improvement_rounded,
                    color: color, size: 20)),
            const SizedBox(width: 12),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(cd.currentPhase,
                      style: TextStyle(
                          color: color,
                          fontSize: 15,
                          fontWeight: FontWeight.w800,
                          decoration: TextDecoration.none)),
                  Text('Started ${_fmt(cd.cycleStartDate)}',
                      style: const TextStyle(
                          color: _textMid,
                          fontSize: 11,
                          decoration: TextDecoration.none)),
                ])),
            // Today's log button
            GestureDetector(
              onTap: () => _showDailyLogSheet(DateTime.now()),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                        color: _pink.withOpacity(.3),
                        blurRadius: 8,
                        offset: const Offset(0, 3))
                  ],
                ),
                child: const Text('+ Log Today',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        decoration: TextDecoration.none)),
              ),
            ),
          ]),
          const SizedBox(height: 14),
          Row(children: [
            _qs('Cycle', '${cd.cycleLength}d', _pink),
            _qs('Period', '${cd.periodDuration}d', _purple),
            _qs('Ovulation', _sd(cd.ovulationDay), _teal),
            _qs('Next', _sd(cd.nextPeriodStart), _amber),
          ]),
          const SizedBox(height: 12),
          // Phase tip
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
                color: color.withOpacity(.06),
                borderRadius: BorderRadius.circular(12)),
            child: Row(children: [
              Icon(Icons.lightbulb_rounded, color: color, size: 15),
              const SizedBox(width: 8),
              Expanded(
                  child: Text(_phaseTip(cd.currentPhase),
                      style: TextStyle(
                          color: _textDark.withOpacity(.7),
                          fontSize: 12,
                          height: 1.4,
                          decoration: TextDecoration.none))),
            ]),
          ),
          // Today's daily log summary if exists
          if (_dailyLogs.containsKey(_todayKey())) ...[
            const SizedBox(height: 10),
            _todayLogSummary(_dailyLogs[_todayKey()]!),
          ],
        ]),
      ),
    );
  }

  Widget _todayLogSummary(DailyLog log) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _pink.withOpacity(.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _pink.withOpacity(.15)),
      ),
      child: Row(children: [
        Text(log.flow.emoji,
            style:
                const TextStyle(fontSize: 16, decoration: TextDecoration.none)),
        const SizedBox(width: 8),
        Expanded(
            child: Wrap(spacing: 4, children: [
          ...log.moods.map((m) => Text(m.emoji,
              style: const TextStyle(
                  fontSize: 14, decoration: TextDecoration.none))),
          if (log.symptoms.isNotEmpty)
            Text('• ${log.symptoms.take(2).join(", ")}',
                style: const TextStyle(
                    color: _textMid,
                    fontSize: 11,
                    decoration: TextDecoration.none)),
        ])),
        Text("Today's log ✓",
            style: TextStyle(
                color: _pink.withOpacity(.7),
                fontSize: 10,
                fontWeight: FontWeight.w700,
                decoration: TextDecoration.none)),
      ]),
    );
  }

  Widget _qs(String label, String val, Color color) => Expanded(
          child: Column(children: [
        Text(val,
            style: TextStyle(
                color: color,
                fontSize: 13,
                fontWeight: FontWeight.w800,
                decoration: TextDecoration.none)),
        const SizedBox(height: 2),
        Text(label,
            style: const TextStyle(
                color: _textMid,
                fontSize: 10,
                decoration: TextDecoration.none)),
      ]));

  // ── Tabs ───────────────────────────────────────────────────────────────────

  static const _tabLabels = [
    'Overview',
    'Calendar',
    'Log',
    'History',
    'Insights'
  ];

  Widget _tabs() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
        child: Container(
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withOpacity(.05),
                  blurRadius: 10,
                  offset: const Offset(0, 2))
            ],
          ),
          child: Row(
              children: List.generate(
                  _tabLabels.length,
                  (i) => Expanded(
                        child: GestureDetector(
                          onTap: () => setState(() => _tab = i),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 200),
                            padding: const EdgeInsets.symmetric(vertical: 10),
                            decoration: BoxDecoration(
                              gradient: _tab == i
                                  ? const LinearGradient(
                                      colors: [_pinkLight, _pink])
                                  : null,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(_tabLabels[i],
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  color: _tab == i ? Colors.white : _textMid,
                                  fontSize: 11,
                                  fontWeight: _tab == i
                                      ? FontWeight.w700
                                      : FontWeight.w500,
                                  decoration: TextDecoration.none,
                                )),
                          ),
                        ),
                      ))),
        ),
      );

  Widget _tabContent() {
    switch (_tab) {
      case 0:
        return _overviewTab();
      case 1:
        return _calendarTab();
      case 2:
        return _dailyLogTab();
      case 3:
        return _historyTab();
      case 4:
        return _insightsTab();
      default:
        return const SizedBox.shrink();
    }
  }

  // ── Overview tab ───────────────────────────────────────────────────────────

  Widget _overviewTab() {
    if (_latest == null) return _emptyState();
    final cd = _latest!;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(children: [
        const SizedBox(height: 12),
        _secTitle('Upcoming Events'),
        const SizedBox(height: 10),
        _eventCard(
            Icons.water_drop_rounded,
            _pink,
            'Next Period',
            _fmt(cd.nextPeriodStart),
            cd.daysUntilNext > 0 ? '${cd.daysUntilNext}d away' : 'Due!'),
        _eventCard(
            Icons.favorite_rounded,
            _purple,
            'Fertile Window',
            '${_sd(cd.fertileDays.first)} – ${_sd(cd.fertileDays.last)}',
            '6 days'),
        _eventCard(Icons.star_rounded, _teal, 'Ovulation Day',
            _fmt(cd.ovulationDay), _ovulCd(cd)),
        _eventCard(Icons.wb_cloudy_rounded, _amber, 'PMS Window',
            '${_sd(cd.pmsDays.first)} – ${_sd(cd.pmsDays.last)}', '5 days'),
        const SizedBox(height: 16),
        _secTitle('Cycle Parameters'),
        const SizedBox(height: 10),
        _paramCard(cd),
      ]),
    );
  }

  Widget _emptyState() => Padding(
        padding: const EdgeInsets.all(40),
        child: Column(children: [
          const Text('📅',
              style: TextStyle(fontSize: 48, decoration: TextDecoration.none)),
          const SizedBox(height: 12),
          const Text('No cycle data yet',
              style: TextStyle(
                  color: _textDark,
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  decoration: TextDecoration.none)),
          const SizedBox(height: 8),
          const Text(
              'Tap + to log your current cycle, or use History to add past cycles.',
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: _textMid,
                  fontSize: 13,
                  height: 1.5,
                  decoration: TextDecoration.none)),
          const SizedBox(height: 24),
          Row(children: [
            Expanded(
                child: _primaryBtn('Log Current Cycle', () => _showLogSheet())),
            const SizedBox(width: 12),
            Expanded(
                child: _outlineBtn(
                    'Add History', () => _showLogSheet(isHistory: true))),
          ]),
        ]),
      );

  Widget _eventCard(
      IconData icon, Color color, String title, String sub, String trail) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(.15), width: 1.5),
        boxShadow: [
          BoxShadow(
              color: color.withOpacity(.05),
              blurRadius: 10,
              offset: const Offset(0, 3))
        ],
      ),
      child: Row(children: [
        Container(
            padding: const EdgeInsets.all(9),
            decoration: BoxDecoration(
                color: color.withOpacity(.12),
                borderRadius: BorderRadius.circular(10)),
            child: Icon(icon, color: color, size: 18)),
        const SizedBox(width: 12),
        Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title,
              style: const TextStyle(
                  color: _textDark,
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  decoration: TextDecoration.none)),
          const SizedBox(height: 2),
          Text(sub,
              style: const TextStyle(
                  color: _textMid,
                  fontSize: 11,
                  decoration: TextDecoration.none)),
        ])),
        _pillChip(trail, color),
      ]),
    );
  }

  Widget _paramCard(CycleData cd) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
        ),
        child: Column(children: [
          _prow('Cycle length', '${cd.cycleLength} days', _pink),
          const Divider(height: 18, color: Color(0xFFF5E6F0)),
          _prow('Period duration', '${cd.periodDuration} days', _purple),
          const Divider(height: 18, color: Color(0xFFF5E6F0)),
          _prow('Cycle started', _fmt(cd.cycleStartDate), _teal),
          const Divider(height: 18, color: Color(0xFFF5E6F0)),
          _prow('Cycles logged', '${_history.length}', _amber),
          const Divider(height: 18, color: Color(0xFFF5E6F0)),
          _prow('Regularity', _regularity.label, _regularity.color),
        ]),
      );

  Widget _prow(String l, String v, Color c) => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(l,
              style: const TextStyle(
                  color: Color(0xFFAA99BB),
                  fontSize: 12,
                  decoration: TextDecoration.none)),
          Text(v,
              style: TextStyle(
                  color: c,
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  decoration: TextDecoration.none)),
        ],
      );

  // ── Calendar tab ───────────────────────────────────────────────────────────

  Widget _calendarTab() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(children: [
          const SizedBox(height: 12),
          _monthNav(),
          const SizedBox(height: 12),
          _calGrid(),
          const SizedBox(height: 12),
          _calLegend(),
          const SizedBox(height: 12),
          // Tap info
          const Text('Tap any period day to log symptoms',
              style: TextStyle(
                  color: _textMid,
                  fontSize: 11,
                  decoration: TextDecoration.none)),
        ]),
      );

  Widget _monthNav() => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _navBtn(
              Icons.chevron_left_rounded,
              () => setState(() =>
                  _calMonth = DateTime(_calMonth.year, _calMonth.month - 1))),
          Text('${_monthName(_calMonth.month)} ${_calMonth.year}',
              style: const TextStyle(
                  color: _textDark,
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  decoration: TextDecoration.none)),
          _navBtn(
              Icons.chevron_right_rounded,
              () => setState(() =>
                  _calMonth = DateTime(_calMonth.year, _calMonth.month + 1))),
        ],
      );

  Widget _navBtn(IconData icon, VoidCallback fn) => GestureDetector(
      onTap: fn,
      child: Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            color: const Color(0xFFFFF0F7),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: const Color(0xFFEEC4D6)),
          ),
          child: Icon(icon, color: _pink, size: 22)));

  Widget _calGrid() {
    // Build sets for the displayed month from ALL history cycles
    final daysInMonth = DateTime(_calMonth.year, _calMonth.month + 1, 0).day;
    final offset = DateTime(_calMonth.year, _calMonth.month, 1).weekday - 1;

    Set<int> periodSet = {}, fertileSet = {}, ovulSet = {}, pmsSet = {};
    Set<int> loggedSet = {};

    void addIfMatch(DateTime d, Set<int> s) {
      if (d.year == _calMonth.year && d.month == _calMonth.month) s.add(d.day);
    }

    for (final cd in _history) {
      for (final d in cd.periodDays) addIfMatch(d, periodSet);
      for (final d in cd.fertileDays) addIfMatch(d, fertileSet);
      addIfMatch(cd.ovulationDay, ovulSet);
      for (final d in cd.pmsDays) addIfMatch(d, pmsSet);
    }

    // Daily logs
    for (final entry in _dailyLogs.entries) {
      try {
        final d = DateTime.parse(entry.key);
        if (d.year == _calMonth.year &&
            d.month == _calMonth.month &&
            entry.value.symptoms.isNotEmpty) loggedSet.add(d.day);
      } catch (_) {}
    }

    final cells = [
      ...List<int?>.filled(offset, null),
      ...List<int?>.generate(daysInMonth, (i) => i + 1),
    ];
    while (cells.length % 7 != 0) cells.add(null);
    final rows = cells.length ~/ 7;

    return Column(children: [
      Row(
          children: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
              .map((d) => Expanded(
                  child: Center(
                      child: Text(d,
                          style: TextStyle(
                              color: (d == 'Sat' || d == 'Sun')
                                  ? const Color(0xFFE087A8)
                                  : _textMid,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              decoration: TextDecoration.none)))))
              .toList()),
      const SizedBox(height: 8),
      ...List.generate(
          rows,
          (row) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                    children: List.generate(7, (col) {
                  final day = cells[row * 7 + col];
                  if (day == null) return const Expanded(child: SizedBox());
                  return Expanded(
                      child: _calDay(day, periodSet, fertileSet, ovulSet,
                          pmsSet, loggedSet, col));
                })),
              )),
    ]);
  }

  Widget _calDay(int day, Set<int> per, Set<int> fer, Set<int> ov, Set<int> pms,
      Set<int> logged, int col) {
    final isPer = per.contains(day);
    final isFer = fer.contains(day);
    final isOv = ov.contains(day);
    final isPms = pms.contains(day);
    final isWknd = col >= 5;
    final isToday = day == DateTime.now().day &&
        _calMonth.month == DateTime.now().month &&
        _calMonth.year == DateTime.now().year;
    final hasLog = logged.contains(day);

    Gradient? grad;
    Color? bg;
    Color txt = isWknd ? const Color(0xFFE087A8) : _textDark;

    if (isPer) {
      grad = const LinearGradient(colors: [_pinkLight, _pink]);
      txt = Colors.white;
    } else if (isOv) {
      grad = const LinearGradient(colors: [Color(0xFF80D8CC), _teal]);
      txt = Colors.white;
    } else if (isFer) {
      grad = const LinearGradient(colors: [Color(0xFFB5A4E0), _purple]);
      txt = Colors.white;
    } else if (isPms) {
      bg = const Color(0xFFFFF3CD);
      txt = _amber;
    } else if (isToday) {
      bg = const Color(0xFFFFD6E8);
      txt = _pink;
    }

    return GestureDetector(
      onTap: () {
        final d = DateTime(_calMonth.year, _calMonth.month, day);
        _showDailyLogSheet(d);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        margin: const EdgeInsets.symmetric(horizontal: 2),
        height: 44,
        decoration: BoxDecoration(
          gradient: grad,
          color: grad == null ? bg : null,
          borderRadius: BorderRadius.circular(12),
          border: isToday && !isPer
              ? Border.all(color: _pinkLight.withOpacity(.5), width: 1.5)
              : null,
          boxShadow: (isPer || isFer || isOv)
              ? [
                  BoxShadow(
                      color: (isPer
                              ? _pink
                              : isFer
                                  ? _purple
                                  : _teal)
                          .withOpacity(.28),
                      blurRadius: 6,
                      offset: const Offset(0, 3))
                ]
              : null,
        ),
        child: Stack(alignment: Alignment.center, children: [
          Text('$day',
              style: TextStyle(
                  color: txt,
                  fontSize: 14,
                  fontWeight: isPer || isFer || isOv || isToday
                      ? FontWeight.w700
                      : FontWeight.w500,
                  decoration: TextDecoration.none)),
          if (hasLog)
            Positioned(
                bottom: 4,
                child: Container(
                    width: 4,
                    height: 4,
                    decoration: const BoxDecoration(
                        color: _pink, shape: BoxShape.circle))),
        ]),
      ),
    );
  }

  Widget _calLegend() => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
            color: const Color(0xFFFFF5F9),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0xFFEED4E0))),
        child: Wrap(
            spacing: 12,
            runSpacing: 6,
            alignment: WrapAlignment.center,
            children: [
              _ldot(_pink, 'Period'),
              _ldot(_purple, 'Fertile'),
              _ldot(_teal, 'Ovulation'),
              _ldot(_amber, 'PMS'),
              _ldot(const Color(0xFFFFD6E8), 'Today'),
            ]),
      );

  Widget _ldot(Color c, String l) =>
      Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
                color: c,
                borderRadius: BorderRadius.circular(3),
                boxShadow: [
                  BoxShadow(color: c.withOpacity(.4), blurRadius: 4)
                ])),
        const SizedBox(width: 4),
        Text(l,
            style: TextStyle(
                color: c,
                fontSize: 10,
                fontWeight: FontWeight.w600,
                decoration: TextDecoration.none)),
      ]);

  // ── Daily Log Tab ──────────────────────────────────────────────────────────

  Widget _dailyLogTab() {
    final today = _todayKey();
    final log = _dailyLogs[today];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(children: [
        const SizedBox(height: 12),
        _secTitle("Today's Log  •  ${_fmt(DateTime.now())}"),
        const SizedBox(height: 12),
        if (log != null) ...[
          _existingLogCard(log),
          const SizedBox(height: 12),
        ],
        _primaryBtn(
          log != null ? '✏️  Edit Today\'s Log' : '+ Log Today',
          () => _showDailyLogSheet(DateTime.now()),
        ),
        const SizedBox(height: 20),
        _secTitle('Recent Daily Logs'),
        const SizedBox(height: 10),
        ..._recentLogs(),
      ]),
    );
  }

  Widget _existingLogCard(DailyLog log) => Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: _pink.withOpacity(.2), width: 1.5),
          boxShadow: [
            BoxShadow(
                color: _pink.withOpacity(.06),
                blurRadius: 10,
                offset: const Offset(0, 3))
          ],
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text(log.flow.emoji,
                style: const TextStyle(
                    fontSize: 20, decoration: TextDecoration.none)),
            const SizedBox(width: 8),
            Text('Flow: ${log.flow.label}',
                style: const TextStyle(
                    color: _textDark,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.none)),
          ]),
          if (log.moods.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
                spacing: 6,
                children: log.moods
                    .map((m) => Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                            color: m.color.withOpacity(.10),
                            borderRadius: BorderRadius.circular(20)),
                        child: Text('${m.emoji} ${m.label}',
                            style: TextStyle(
                                color: m.color,
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                decoration: TextDecoration.none))))
                    .toList()),
          ],
          if (log.symptoms.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
                spacing: 6,
                runSpacing: 4,
                children: log.symptoms.map((s) {
                  final idx = kSymptoms.indexOf(s);
                  return Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                          color: _pink.withOpacity(.08),
                          borderRadius: BorderRadius.circular(20)),
                      child: Text('${idx >= 0 ? kSymptomEmojis[idx] : "•"} $s',
                          style: const TextStyle(
                              color: _pink,
                              fontSize: 11,
                              decoration: TextDecoration.none)));
                }).toList()),
          ],
          if (log.note.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text('📝 ${log.note}',
                style: const TextStyle(
                    color: _textMid,
                    fontSize: 12,
                    decoration: TextDecoration.none)),
          ],
        ]),
      );

  List<Widget> _recentLogs() {
    final sorted = _dailyLogs.entries.toList()
      ..sort((a, b) => b.key.compareTo(a.key));
    if (sorted.isEmpty)
      return [
        const Center(
            child: Padding(
                padding: EdgeInsets.all(20),
                child: Text('No logs yet',
                    style: TextStyle(
                        color: _textMid,
                        fontSize: 13,
                        decoration: TextDecoration.none))))
      ];
    return sorted.take(7).map((e) {
      final d = DateTime.tryParse(e.key) ?? DateTime.now();
      final log = e.value;
      return Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0xFFFCE7F3), width: 1.2)),
        child: Row(children: [
          Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                  color: _pink.withOpacity(.08),
                  borderRadius: BorderRadius.circular(10)),
              child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('${d.day}',
                        style: const TextStyle(
                            color: _pink,
                            fontSize: 15,
                            fontWeight: FontWeight.w900,
                            decoration: TextDecoration.none)),
                    Text(_sm(d.month),
                        style: const TextStyle(
                            color: Color(0xFFD4A0B8),
                            fontSize: 9,
                            decoration: TextDecoration.none)),
                  ])),
          const SizedBox(width: 10),
          Expanded(
              child: Wrap(spacing: 4, runSpacing: 2, children: [
            Text(log.flow.emoji,
                style: const TextStyle(
                    fontSize: 16, decoration: TextDecoration.none)),
            ...log.moods.map((m) => Text(m.emoji,
                style: const TextStyle(
                    fontSize: 14, decoration: TextDecoration.none))),
            if (log.symptoms.isNotEmpty)
              Text(
                  '${log.symptoms.length} symptom${log.symptoms.length > 1 ? "s" : ""}',
                  style: const TextStyle(
                      color: _textMid,
                      fontSize: 11,
                      decoration: TextDecoration.none)),
          ])),
          GestureDetector(
              onTap: () => _showDailyLogSheet(d),
              child: const Icon(Icons.edit_rounded, color: _textMid, size: 16)),
        ]),
      );
    }).toList();
  }

  // ── History tab ────────────────────────────────────────────────────────────

  Widget _historyTab() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(children: [
          const SizedBox(height: 12),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            _secTitle('Cycle History (${_history.length})'),
            GestureDetector(
                onTap: () => _showLogSheet(isHistory: true),
                child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                        gradient:
                            const LinearGradient(colors: [_pinkLight, _pink]),
                        borderRadius: BorderRadius.circular(20)),
                    child: const Text('+ Add Past Cycle',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none)))),
          ]),
          const SizedBox(height: 10),
          if (_history.isEmpty)
            const Padding(
                padding: EdgeInsets.all(30),
                child: Text(
                    'No cycles logged yet.\nTap "+ Add Past Cycle" to begin.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                        color: _textMid,
                        fontSize: 13,
                        height: 1.5,
                        decoration: TextDecoration.none)))
          else
            ..._history.map((cd) => _historyCard(cd)).toList(),
        ]),
      );

  Widget _historyCard(CycleData cd) {
    final isLatest =
        _latest != null && _sameDay(cd.cycleStartDate, _latest!.cycleStartDate);
    return Dismissible(
      key: Key(cd.cycleStartDate.toIso8601String()),
      direction: DismissDirection.endToStart,
      confirmDismiss: (_) => _confirmDialog(
          'Delete Cycle', 'Remove cycle starting ${_fmt(cd.cycleStartDate)}?'),
      onDismissed: (_) => _deleteCycle(cd),
      background: Container(
          margin: const EdgeInsets.only(bottom: 10),
          decoration: BoxDecoration(
              color: _pink.withOpacity(.1),
              borderRadius: BorderRadius.circular(16)),
          alignment: Alignment.centerRight,
          padding: const EdgeInsets.only(right: 20),
          child: const Icon(Icons.delete_rounded, color: _pink)),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color:
                  isLatest ? _pink.withOpacity(.35) : const Color(0xFFFCE7F3),
              width: 1.5),
          boxShadow: [
            BoxShadow(
                color: _pink.withOpacity(.05),
                blurRadius: 10,
                offset: const Offset(0, 3))
          ],
        ),
        child: Row(children: [
          Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(
                  color: isLatest
                      ? _pink.withOpacity(.12)
                      : const Color(0xFFFFF0F7),
                  borderRadius: BorderRadius.circular(12)),
              child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('${cd.cycleStartDate.day}',
                        style: TextStyle(
                            color: isLatest ? _pink : const Color(0xFFC85A7A),
                            fontSize: 16,
                            fontWeight: FontWeight.w900,
                            decoration: TextDecoration.none)),
                    Text(_sm(cd.cycleStartDate.month),
                        style: const TextStyle(
                            color: Color(0xFFD4A0B8),
                            fontSize: 10,
                            decoration: TextDecoration.none)),
                  ])),
          const SizedBox(width: 12),
          Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Row(children: [
                  Text(_fmt(cd.cycleStartDate),
                      style: const TextStyle(
                          color: _textDark,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          decoration: TextDecoration.none)),
                  const SizedBox(width: 6),
                  if (isLatest) _pillChip('Current', _pink),
                  if (cd.isHistorical) _pillChip('History', _purple),
                ]),
                const SizedBox(height: 2),
                Text(
                    'Cycle: ${cd.cycleLength}d  •  Period: ${cd.periodDuration}d  •  Next: ${_sd(cd.nextPeriodStart)}',
                    style: const TextStyle(
                        color: _textMid,
                        fontSize: 11,
                        decoration: TextDecoration.none)),
              ])),
          // Edit button
          GestureDetector(
              onTap: () => _showLogSheet(edit: cd),
              child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                      color: const Color(0xFFFFF0F7),
                      borderRadius: BorderRadius.circular(10)),
                  child:
                      const Icon(Icons.edit_rounded, color: _pink, size: 16))),
        ]),
      ),
    );
  }

  // ── Insights tab ───────────────────────────────────────────────────────────

  Widget _insightsTab() {
    if (_history.isEmpty)
      return Padding(
          padding: const EdgeInsets.all(40),
          child: Column(children: const [
            Text('📊',
                style:
                    TextStyle(fontSize: 48, decoration: TextDecoration.none)),
            SizedBox(height: 12),
            Text('Log a few cycles to see insights',
                textAlign: TextAlign.center,
                style: TextStyle(
                    color: _textMid,
                    fontSize: 13,
                    decoration: TextDecoration.none)),
          ]));

    final lens = _history.map((e) => e.cycleLength).toList();
    final avg = lens.reduce((a, b) => a + b) / lens.length;
    final avgPd =
        _history.map((e) => e.periodDuration).reduce((a, b) => a + b) /
            _history.length;

    // Symptom frequency from daily logs
    final Map<String, int> symCount = {};
    for (final log in _dailyLogs.values) {
      for (final s in log.symptoms) symCount[s] = (symCount[s] ?? 0) + 1;
    }
    final topSymptoms = symCount.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    // Mood frequency
    final Map<String, int> moodCount = {};
    for (final log in _dailyLogs.values) {
      for (final m in log.moods) {
        moodCount[m.label] = (moodCount[m.label] ?? 0) + 1;
      }
    }
    final topMoods = moodCount.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(children: [
        const SizedBox(height: 12),
        _secTitle('Cycle Statistics'),
        const SizedBox(height: 10),
        Row(children: [
          Expanded(
              child: _statTile(avg.toStringAsFixed(1), 'Avg Cycle', _pink)),
          const SizedBox(width: 10),
          Expanded(
              child:
                  _statTile(avgPd.toStringAsFixed(1), 'Avg Period', _purple)),
          const SizedBox(width: 10),
          Expanded(child: _statTile('${_history.length}', 'Logged', _teal)),
        ]),
        const SizedBox(height: 16),

        // Trend chart
        _secTitle('Cycle Length Trend'),
        const SizedBox(height: 10),
        _trendChart(),
        const SizedBox(height: 16),

        // Regularity insight
        _insightCard(
          _regularity.icon,
          _regularity.color,
          _regularity.label,
          _regularity == Regularity.regular
              ? 'Your cycles are consistent! Variations of ≤3 days are normal.'
              : _regularity == Regularity.slightlyIrregular
                  ? 'Cycles vary by ${lens.reduce(math.max) - lens.reduce(math.min)} days. Mild irregularity may be normal.'
                  : 'Significant cycle variability detected. Consider discussing with your doctor.',
        ),
        const SizedBox(height: 10),

        if (topSymptoms.isNotEmpty) ...[
          _secTitle('Top Reported Symptoms'),
          const SizedBox(height: 10),
          _symptomFreq(topSymptoms.take(5).toList()),
          const SizedBox(height: 16),
        ],

        if (topMoods.isNotEmpty) ...[
          _secTitle('Mood Patterns'),
          const SizedBox(height: 10),
          Wrap(
              spacing: 8,
              runSpacing: 6,
              children: topMoods.take(4).map((e) {
                final mood = Mood.values.firstWhere((m) => m.label == e.key,
                    orElse: () => Mood.calm);
                return Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                        color: mood.color.withOpacity(.10),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: mood.color.withOpacity(.25))),
                    child: Text('${mood.emoji} ${mood.label} (${e.value}×)',
                        style: TextStyle(
                            color: mood.color,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            decoration: TextDecoration.none)));
              }).toList()),
          const SizedBox(height: 16),
        ],

        _insightCard(Icons.monitor_heart_rounded, _pink, 'Health Connection',
            'Your cycle logs improve PCOD, endometriosis & cervical cancer prediction accuracy.'),
        const SizedBox(height: 10),
        _insightCard(Icons.analytics_rounded, _purple, 'Prediction Confidence',
            '${(_predictionConfidence * 100).toInt()}% — ${_predictionConfidence >= .80 ? "High accuracy based on ${_history.length} cycles." : "Log more cycles to improve accuracy."}'),
      ]),
    );
  }

  Widget _trendChart() {
    final data = _history.reversed.toList().take(6).toList();
    if (data.length < 2)
      return const Padding(
          padding: EdgeInsets.all(12),
          child: Text('Log at least 2 cycles to see the trend.',
              style: TextStyle(
                  color: _textMid,
                  fontSize: 12,
                  decoration: TextDecoration.none)));
    final maxLen = data.map((e) => e.cycleLength).reduce(math.max).toDouble();
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
      child: Column(children: [
        SizedBox(
          height: 100,
          child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: data.map((cd) {
                final h = (cd.cycleLength / maxLen) * 80;
                final isLatest =
                    _sameDay(cd.cycleStartDate, data.last.cycleStartDate);
                return Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Text('${cd.cycleLength}',
                          style: TextStyle(
                              color: isLatest ? _pink : _textMid,
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              decoration: TextDecoration.none)),
                      const SizedBox(height: 3),
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 600),
                        width: 28,
                        height: h,
                        decoration: BoxDecoration(
                            gradient: LinearGradient(
                                begin: Alignment.bottomCenter,
                                end: Alignment.topCenter,
                                colors: isLatest
                                    ? [_pink, _pinkLight]
                                    : [
                                        _purple.withOpacity(.5),
                                        _purple.withOpacity(.25)
                                      ]),
                            borderRadius: const BorderRadius.vertical(
                                top: Radius.circular(6))),
                      ),
                      const SizedBox(height: 4),
                      Text(_sm(cd.cycleStartDate.month),
                          style: const TextStyle(
                              color: _textMid,
                              fontSize: 9,
                              decoration: TextDecoration.none)),
                    ]);
              }).toList()),
        ),
        const SizedBox(height: 8),
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          _ldot(_pink, 'Current'),
          const SizedBox(width: 12),
          _ldot(_purple.withOpacity(.5), 'Past cycles'),
        ]),
      ]),
    );
  }

  Widget _symptomFreq(List<MapEntry<String, int>> data) {
    final max = data.first.value;
    return Column(
        children: data.map((e) {
      final idx = kSymptoms.indexOf(e.key);
      return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(children: [
            Text(idx >= 0 ? kSymptomEmojis[idx] : '•',
                style: const TextStyle(
                    fontSize: 14, decoration: TextDecoration.none)),
            const SizedBox(width: 8),
            SizedBox(
                width: 90,
                child: Text(e.key,
                    style: const TextStyle(
                        color: _textDark,
                        fontSize: 12,
                        decoration: TextDecoration.none))),
            Expanded(
                child: Stack(children: [
              Container(
                  height: 8,
                  decoration: BoxDecoration(
                      color: _pink.withOpacity(.08),
                      borderRadius: BorderRadius.circular(4))),
              FractionallySizedBox(
                  widthFactor: e.value / max,
                  child: Container(
                      height: 8,
                      decoration: BoxDecoration(
                          gradient:
                              const LinearGradient(colors: [_pinkLight, _pink]),
                          borderRadius: BorderRadius.circular(4)))),
            ])),
            const SizedBox(width: 8),
            Text('${e.value}×',
                style: const TextStyle(
                    color: _pink,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.none)),
          ]));
    }).toList());
  }

  Widget _statTile(String v, String l, Color c) => Container(
      padding: const EdgeInsets.symmetric(vertical: 14),
      decoration: BoxDecoration(
          color: c.withOpacity(.08),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: c.withOpacity(.2))),
      child: Column(children: [
        Text(v,
            style: TextStyle(
                color: c,
                fontSize: 20,
                fontWeight: FontWeight.w900,
                decoration: TextDecoration.none)),
        const SizedBox(height: 4),
        Text(l,
            style: const TextStyle(
                color: _textMid,
                fontSize: 11,
                decoration: TextDecoration.none)),
      ]));

  Widget _insightCard(IconData icon, Color color, String title, String body) =>
      Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withOpacity(.15), width: 1.5),
            boxShadow: [
              BoxShadow(
                  color: color.withOpacity(.05),
                  blurRadius: 10,
                  offset: const Offset(0, 3))
            ],
          ),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    color: color.withOpacity(.12),
                    borderRadius: BorderRadius.circular(10)),
                child: Icon(icon, color: color, size: 18)),
            const SizedBox(width: 12),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(title,
                      style: const TextStyle(
                          color: _textDark,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          decoration: TextDecoration.none)),
                  const SizedBox(height: 4),
                  Text(body,
                      style: const TextStyle(
                          color: _textMid,
                          fontSize: 12,
                          height: 1.4,
                          decoration: TextDecoration.none)),
                ])),
          ]));

  // ── Log cycle bottom sheet ─────────────────────────────────────────────────

  Future<void> _showLogSheet({bool isHistory = false, CycleData? edit}) async {
    final vals = await _qaValues();
    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _LogSheet(
        isHistory: isHistory,
        edit: edit,
        defaultCycleLen: vals.cycleLen,
        defaultPeriodDur: vals.periodDur,
        existingDates: _history.map((e) => e.cycleStartDate).toList(),
        onSave: (cd) => _saveCycle(cd, isEdit: edit != null),
      ),
    );
  }

  // ── Daily log bottom sheet ─────────────────────────────────────────────────

  Future<void> _showDailyLogSheet(DateTime date) async {
    final key = _dateKey(date);
    final existing = _dailyLogs[key];
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _DailyLogSheet(
        date: date,
        existing: existing,
        onSave: _saveDailyLog,
      ),
    );
  }

  // ── Notification panel ─────────────────────────────────────────────────────

  void _showNotifPanel() {
    setState(() => _hasUnread = false);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _NotifPanel(
        cd: _latest,
        p: _nPeriod,
        f: _nFertile,
        m: _nMed,
        i: _nInsights,
        onChanged: (p, f, m, i) => setState(() {
          _nPeriod = p;
          _nFertile = f;
          _nMed = m;
          _nInsights = i;
        }),
      ),
    );
  }

  // ── Utilities ──────────────────────────────────────────────────────────────

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg,
          style: const TextStyle(
              fontWeight: FontWeight.w600, color: Colors.white)),
      backgroundColor: _pink,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      margin: const EdgeInsets.all(16),
      duration: const Duration(seconds: 3),
    ));
  }

  Widget _secTitle(String t) => Align(
      alignment: Alignment.centerLeft,
      child: Text(t,
          style: const TextStyle(
              color: _textDark,
              fontSize: 14,
              fontWeight: FontWeight.w800,
              decoration: TextDecoration.none)));

  Widget _pillChip(String t, Color c) => Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
          color: c.withOpacity(.10), borderRadius: BorderRadius.circular(20)),
      child: Text(t,
          style: TextStyle(
              color: c,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              decoration: TextDecoration.none)));

  Widget _primaryBtn(String l, VoidCallback fn) => GestureDetector(
      onTap: fn,
      child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 15),
          decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_pinkLight, _pink]),
              borderRadius: BorderRadius.circular(18),
              boxShadow: [
                BoxShadow(
                    color: _pink.withOpacity(.35),
                    blurRadius: 14,
                    offset: const Offset(0, 5))
              ]),
          child: Center(
              child: Text(l,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      decoration: TextDecoration.none)))));

  Widget _outlineBtn(String l, VoidCallback fn) => GestureDetector(
      onTap: fn,
      child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 15),
          decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: _pink, width: 1.5)),
          child: Center(
              child: Text(l,
                  style: const TextStyle(
                      color: _pink,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      decoration: TextDecoration.none)))));

  Color _phaseColor(String phase) {
    switch (phase) {
      case 'Menstrual Phase':
        return _pink;
      case 'Follicular Phase':
        return _teal;
      case 'Ovulatory Phase':
        return _purple;
      default:
        return _amber;
    }
  }

  String _phaseTip(String phase) {
    switch (phase) {
      case 'Menstrual Phase':
        return 'Rest and stay hydrated. Heat pads help with cramps.';
      case 'Follicular Phase':
        return 'Great energy for new activities! Estrogen is rising.';
      case 'Ovulatory Phase':
        return 'Peak fertility. Best time for conception if planned.';
      default:
        return 'Prioritise sleep and self-care. PMS symptoms may appear.';
    }
  }

  String _ovulCd(CycleData cd) {
    final d = cd.ovulationDay.difference(DateTime.now()).inDays;
    if (d > 0) return 'In ${d}d';
    if (d == 0) return 'Today!';
    return '${-d}d ago';
  }

  String _fmt(DateTime d) => '${d.day} ${_monthName(d.month)} ${d.year}';
  String _sd(DateTime d) => '${d.day} ${_sm(d.month)}';
  String _todayKey() => _dateKey(DateTime.now());
  String _dateKey(DateTime d) =>
      '${d.year}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';
  bool _sameDay(DateTime a, DateTime b) =>
      a.year == b.year && a.month == b.month && a.day == b.day;

  String _monthName(int m) => const [
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
      ][m - 1];
  String _sm(int m) => const [
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
      ][m - 1];
}

// ═══════════════════════════════════════════════════════════════════════════
// LOG CYCLE SHEET  (current + historical + edit)
// ═══════════════════════════════════════════════════════════════════════════

class _LogSheet extends StatefulWidget {
  final bool isHistory;
  final CycleData? edit;
  final int defaultCycleLen;
  final int defaultPeriodDur;
  final List<DateTime> existingDates;
  final void Function(CycleData) onSave;

  const _LogSheet({
    required this.isHistory,
    required this.defaultCycleLen,
    required this.defaultPeriodDur,
    required this.existingDates,
    required this.onSave,
    this.edit,
  });

  @override
  State<_LogSheet> createState() => _LogSheetState();
}

class _LogSheetState extends State<_LogSheet> {
  late DateTime _date;
  late int _cl, _pd;
  bool _isHist = false;

  @override
  void initState() {
    super.initState();
    if (widget.edit != null) {
      _date = widget.edit!.cycleStartDate;
      _cl = widget.edit!.cycleLength;
      _pd = widget.edit!.periodDuration;
      _isHist = widget.edit!.isHistorical;
    } else {
      _date = DateTime.now();
      _cl = widget.defaultCycleLen;
      _pd = widget.defaultPeriodDur;
      _isHist = widget.isHistory;
    }
  }

  bool get _isEdit => widget.edit != null;

  @override
  Widget build(BuildContext context) {
    final preview = CycleData(
        cycleStartDate: _date,
        cycleLength: _cl,
        periodDuration: _pd,
        isHistorical: _isHist);

    return Container(
      decoration: const BoxDecoration(
          color: Color(0xFFFFF5F8),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(
          20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _handle(),
        // Title
        Row(children: [
          Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                  borderRadius: BorderRadius.circular(12)),
              child: Icon(
                  _isEdit ? Icons.edit_rounded : Icons.water_drop_rounded,
                  color: Colors.white,
                  size: 20)),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(
                _isEdit
                    ? 'Edit Cycle'
                    : _isHist
                        ? 'Add Past Cycle'
                        : 'Log Current Cycle',
                style: const TextStyle(
                    color: _textDark,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    decoration: TextDecoration.none)),
            Text(
                _isHist
                    ? 'Enter a past cycle start date'
                    : 'When did this period start?',
                style: const TextStyle(
                    color: _textMid,
                    fontSize: 12,
                    decoration: TextDecoration.none)),
          ]),
        ]),
        const SizedBox(height: 24),

        // History / Current toggle (only when adding new)
        if (!_isEdit) ...[
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFFFCE7F3))),
            child: Row(children: [
              Expanded(
                  child: GestureDetector(
                      onTap: () => setState(() => _isHist = false),
                      child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          decoration: BoxDecoration(
                              gradient: !_isHist
                                  ? const LinearGradient(
                                      colors: [_pinkLight, _pink])
                                  : null,
                              borderRadius: BorderRadius.circular(10)),
                          child: Center(
                              child: Text('Current Cycle',
                                  style: TextStyle(
                                      color: !_isHist ? Colors.white : _textMid,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w700,
                                      decoration: TextDecoration.none)))))),
              Expanded(
                  child: GestureDetector(
                      onTap: () => setState(() => _isHist = true),
                      child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          decoration: BoxDecoration(
                              gradient: _isHist
                                  ? const LinearGradient(
                                      colors: [_pinkLight, _pink])
                                  : null,
                              borderRadius: BorderRadius.circular(10)),
                          child: Center(
                              child: Text('Past Cycle',
                                  style: TextStyle(
                                      color: _isHist ? Colors.white : _textMid,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w700,
                                      decoration: TextDecoration.none)))))),
            ]),
          ),
          const SizedBox(height: 20),
        ],

        // Date picker
        const _Lbl('Period Start Date'),
        const SizedBox(height: 8),
        GestureDetector(
          onTap: () async {
            final latest = widget.existingDates.isNotEmpty
                ? widget.existingDates.reduce((a, b) => a.isBefore(b) ? a : b)
                : DateTime(2020);
            final first = _isHist ? DateTime(2018) : latest;
            final last = _isHist
                ? DateTime.now().subtract(const Duration(days: 1))
                : DateTime.now();
            final p = await showDatePicker(
                context: context,
                initialDate: _date.isAfter(last) ? last : _date,
                firstDate: first,
                lastDate: last,
                builder: (ctx, child) => Theme(
                    data: Theme.of(ctx).copyWith(
                        colorScheme: const ColorScheme.light(
                            primary: _pink,
                            onPrimary: Colors.white,
                            surface: Colors.white)),
                    child: child!));
            if (p != null) setState(() => _date = p);
          },
          child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: _pink.withOpacity(.3), width: 1.5)),
              child: Row(children: [
                const Icon(Icons.calendar_today_rounded,
                    color: _pink, size: 20),
                const SizedBox(width: 12),
                Text('${_date.day} ${_mn(_date.month)} ${_date.year}',
                    style: const TextStyle(
                        color: _textDark,
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        decoration: TextDecoration.none)),
                const Spacer(),
                const Icon(Icons.chevron_right_rounded,
                    color: Color(0xFFD4A0B8)),
              ])),
        ),
        const SizedBox(height: 20),

        // Info banner
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
              color: _purple.withOpacity(.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _purple.withOpacity(.25))),
          child: const Row(children: [
            Icon(Icons.info_outline_rounded, color: _purple, size: 16),
            SizedBox(width: 8),
            Expanded(
                child: Text(
                    'Cycle length & period duration are pre-filled from your questionnaire.',
                    style: TextStyle(
                        color: _purple,
                        fontSize: 11,
                        height: 1.4,
                        decoration: TextDecoration.none))),
          ]),
        ),
        const SizedBox(height: 20),

        const _Lbl('Cycle Length (days)'),
        const SizedBox(height: 8),
        _StepperWidget(
            value: _cl,
            min: 21,
            max: 45,
            onChanged: (v) => setState(() => _cl = v)),
        const SizedBox(height: 16),

        const _Lbl('Period Duration (days)'),
        const SizedBox(height: 8),
        _StepperWidget(
            value: _pd,
            min: 1,
            max: 10,
            onChanged: (v) => setState(() => _pd = v)),
        const SizedBox(height: 24),

        // Preview
        _preview(preview),
        const SizedBox(height: 24),

        GestureDetector(
            onTap: () {
              widget.onSave(CycleData(
                id: widget.edit?.id,
                cycleStartDate: _date,
                cycleLength: _cl,
                periodDuration: _pd,
                isHistorical: _isHist,
              ));
              Navigator.pop(context);
            },
            child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16),
                decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                    borderRadius: BorderRadius.circular(18),
                    boxShadow: [
                      BoxShadow(
                          color: _pink.withOpacity(.35),
                          blurRadius: 14,
                          offset: const Offset(0, 5))
                    ]),
                child: Center(
                    child: Text(_isEdit ? 'Update Cycle' : 'Save Cycle',
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none))))),
      ])),
    );
  }

  Widget _preview(CycleData cd) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('🔮 Prediction Preview',
              style: TextStyle(
                  color: _textDark,
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  decoration: TextDecoration.none)),
          const SizedBox(height: 12),
          _pr('Next Period',
              '${cd.nextPeriodStart.day} ${_sm(cd.nextPeriodStart.month)} ${cd.nextPeriodStart.year}'),
          _pr('Ovulation',
              '${cd.ovulationDay.day} ${_sm(cd.ovulationDay.month)}'),
          _pr('Fertile Window',
              '${cd.fertileDays.first.day}–${cd.fertileDays.last.day} ${_sm(cd.fertileDays.last.month)}'),
          _pr('PMS Window',
              '${cd.pmsDays.first.day}–${cd.pmsDays.last.day} ${_sm(cd.pmsDays.last.month)}'),
        ]),
      );

  Widget _pr(String l, String v) => Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(l,
            style: const TextStyle(
                color: Color(0xFFAA99BB),
                fontSize: 12,
                decoration: TextDecoration.none)),
        Text(v,
            style: const TextStyle(
                color: _pink,
                fontSize: 12,
                fontWeight: FontWeight.w700,
                decoration: TextDecoration.none)),
      ]));

  String _mn(int m) => const [
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
      ][m - 1];
  String _sm(int m) => const [
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
      ][m - 1];
}

// ═══════════════════════════════════════════════════════════════════════════
// DAILY LOG SHEET
// ═══════════════════════════════════════════════════════════════════════════

class _DailyLogSheet extends StatefulWidget {
  final DateTime date;
  final DailyLog? existing;
  final void Function(DailyLog) onSave;

  const _DailyLogSheet(
      {required this.date, required this.onSave, this.existing});

  @override
  State<_DailyLogSheet> createState() => _DailyLogSheetState();
}

class _DailyLogSheetState extends State<_DailyLogSheet> {
  late FlowIntensity _flow;
  late List<Mood> _moods;
  late List<String> _symptoms;
  final _noteCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _flow = widget.existing?.flow ?? FlowIntensity.none;
    _moods = List.from(widget.existing?.moods ?? []);
    _symptoms = List.from(widget.existing?.symptoms ?? []);
    _noteCtrl.text = widget.existing?.note ?? '';
  }

  @override
  void dispose() {
    _noteCtrl.dispose();
    super.dispose();
  }

  String _mn(int m) => const [
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
      ][m - 1];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
          color: Color(0xFFFFF5F8),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(
          20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _handle(),
        Row(children: [
          Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                  borderRadius: BorderRadius.circular(12)),
              child: const Icon(Icons.edit_note_rounded,
                  color: Colors.white, size: 20)),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Daily Log',
                style: TextStyle(
                    color: _textDark,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    decoration: TextDecoration.none)),
            Text(
                '${widget.date.day} ${_mn(widget.date.month)} ${widget.date.year}',
                style: const TextStyle(
                    color: _textMid,
                    fontSize: 12,
                    decoration: TextDecoration.none)),
          ]),
        ]),
        const SizedBox(height: 24),

        // Flow
        const _Lbl('Flow Intensity'),
        const SizedBox(height: 10),
        Row(
          children: FlowIntensity.values
              .map((f) => Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _flow = f),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        margin: const EdgeInsets.symmetric(horizontal: 3),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        decoration: BoxDecoration(
                            color: _flow == f ? _pink : Colors.white,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                                color: _flow == f
                                    ? _pink
                                    : const Color(0xFFFCE7F3),
                                width: 1.5)),
                        child: Column(children: [
                          Text(f.emoji,
                              style: const TextStyle(
                                  fontSize: 16,
                                  decoration: TextDecoration.none)),
                          const SizedBox(height: 3),
                          Text(f.label,
                              style: TextStyle(
                                  color: _flow == f ? Colors.white : _textMid,
                                  fontSize: 9,
                                  fontWeight: FontWeight.w600,
                                  decoration: TextDecoration.none)),
                        ]),
                      ),
                    ),
                  ))
              .toList(),
        ),
        const SizedBox(height: 20),

        // Mood
        const _Lbl('Mood'),
        const SizedBox(height: 10),
        Wrap(
            spacing: 8,
            runSpacing: 8,
            children: Mood.values.map((m) {
              final sel = _moods.contains(m);
              return GestureDetector(
                  onTap: () =>
                      setState(() => sel ? _moods.remove(m) : _moods.add(m)),
                  child: AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                          color: sel ? m.color : Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                              color: sel ? m.color : m.color.withOpacity(.3),
                              width: 1.5)),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Text(m.emoji,
                            style: const TextStyle(
                                fontSize: 14, decoration: TextDecoration.none)),
                        const SizedBox(width: 5),
                        Text(m.label,
                            style: TextStyle(
                                color: sel ? Colors.white : m.color,
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                decoration: TextDecoration.none)),
                      ])));
            }).toList()),
        const SizedBox(height: 20),

        // Symptoms
        const _Lbl('Symptoms'),
        const SizedBox(height: 10),
        Wrap(
            spacing: 8,
            runSpacing: 8,
            children: List.generate(kSymptoms.length, (i) {
              final s = kSymptoms[i];
              final em = kSymptomEmojis[i];
              final sel = _symptoms.contains(s);
              return GestureDetector(
                  onTap: () => setState(
                      () => sel ? _symptoms.remove(s) : _symptoms.add(s)),
                  child: AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                          color: sel ? _pink.withOpacity(.12) : Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                              color: sel ? _pink : const Color(0xFFFCE7F3),
                              width: 1.5)),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Text(em,
                            style: const TextStyle(
                                fontSize: 14, decoration: TextDecoration.none)),
                        const SizedBox(width: 5),
                        Text(s,
                            style: TextStyle(
                                color: sel ? _pink : _textMid,
                                fontSize: 12,
                                fontWeight:
                                    sel ? FontWeight.w700 : FontWeight.w500,
                                decoration: TextDecoration.none)),
                      ])));
            })),
        const SizedBox(height: 20),

        // Note
        const _Lbl('Note (optional)'),
        const SizedBox(height: 8),
        Container(
            decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
            child: TextField(
                controller: _noteCtrl,
                maxLines: 3,
                decoration: const InputDecoration(
                    hintText: 'How are you feeling today?',
                    hintStyle: TextStyle(color: _textMid, fontSize: 13),
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.all(14)),
                style: const TextStyle(color: _textDark, fontSize: 13))),
        const SizedBox(height: 24),

        GestureDetector(
            onTap: () {
              final key =
                  '${widget.date.year}-${widget.date.month.toString().padLeft(2, '0')}-${widget.date.day.toString().padLeft(2, '0')}';
              widget.onSave(DailyLog(
                  date: key,
                  flow: _flow,
                  moods: _moods,
                  symptoms: _symptoms,
                  note: _noteCtrl.text.trim()));
              Navigator.pop(context);
            },
            child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16),
                decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                    borderRadius: BorderRadius.circular(18),
                    boxShadow: [
                      BoxShadow(
                          color: _pink.withOpacity(.35),
                          blurRadius: 14,
                          offset: const Offset(0, 5))
                    ]),
                child: const Center(
                    child: Text('Save Log',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none))))),
      ])),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// NOTIFICATION PANEL
// ═══════════════════════════════════════════════════════════════════════════

class _NotifPanel extends StatefulWidget {
  final CycleData? cd;
  final bool p, f, m, i;
  final void Function(bool, bool, bool, bool) onChanged;
  const _NotifPanel(
      {required this.cd,
      required this.p,
      required this.f,
      required this.m,
      required this.i,
      required this.onChanged});
  @override
  State<_NotifPanel> createState() => _NotifPanelState();
}

class _NotifPanelState extends State<_NotifPanel> {
  late bool _p, _f, _m, _i;
  @override
  void initState() {
    super.initState();
    _p = widget.p;
    _f = widget.f;
    _m = widget.m;
    _i = widget.i;
  }

  @override
  Widget build(BuildContext context) {
    final cd = widget.cd;
    return Container(
      decoration: const BoxDecoration(
          color: Color(0xFFFFF5F8),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(
          20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _handle(),
        Row(children: [
          Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                  borderRadius: BorderRadius.circular(12)),
              child: const Icon(Icons.notifications_rounded,
                  color: Colors.white, size: 20)),
          const SizedBox(width: 12),
          const Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Text('Notifications',
                    style: TextStyle(
                        color: _textDark,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        decoration: TextDecoration.none)),
                Text('Reminders & alerts',
                    style: TextStyle(
                        color: _textMid,
                        fontSize: 12,
                        decoration: TextDecoration.none)),
              ])),
          GestureDetector(
              onTap: () => Navigator.pop(context),
              child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                      color: const Color(0xFFF5EEF5),
                      borderRadius: BorderRadius.circular(10)),
                  child: const Icon(Icons.close_rounded,
                      color: Color(0xFFBB8FAE), size: 18))),
        ]),
        if (cd != null) ...[
          const SizedBox(height: 20),
          const Text('Upcoming',
              style: TextStyle(
                  color: _textDark,
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                  decoration: TextDecoration.none)),
          const SizedBox(height: 10),
          _upcoming(
              Icons.water_drop_rounded,
              _pink,
              'Next Period',
              '${cd.nextPeriodStart.day} ${_sm(cd.nextPeriodStart.month)}',
              cd.daysUntilNext > 0 ? 'In ${cd.daysUntilNext}d' : 'Due!'),
          _upcoming(
              Icons.favorite_rounded,
              _purple,
              'Fertile Window',
              '${cd.fertileDays.first.day}–${cd.fertileDays.last.day}',
              '6 days'),
          _upcoming(
              Icons.star_rounded,
              _teal,
              'Ovulation',
              '${cd.ovulationDay.day} ${_sm(cd.ovulationDay.month)}',
              'Approaching'),
        ],
        const SizedBox(height: 20),
        const Text('Settings',
            style: TextStyle(
                color: _textDark,
                fontSize: 14,
                fontWeight: FontWeight.w800,
                decoration: TextDecoration.none)),
        const SizedBox(height: 10),
        _tog(
            'Period Reminder',
            'Alert 2 days before expected period',
            Icons.water_drop_rounded,
            _pink,
            _p,
            (v) => setState(() {
                  _p = v;
                  widget.onChanged(_p, _f, _m, _i);
                })),
        _tog(
            'Fertile Window',
            'Notify when fertile days approach',
            Icons.favorite_rounded,
            _purple,
            _f,
            (v) => setState(() {
                  _f = v;
                  widget.onChanged(_p, _f, _m, _i);
                })),
        _tog(
            'Medicine Reminder',
            'Daily supplement reminder',
            Icons.medication_rounded,
            _teal,
            _m,
            (v) => setState(() {
                  _m = v;
                  widget.onChanged(_p, _f, _m, _i);
                })),
        _tog(
            'Cycle Insights',
            'Weekly cycle health summary',
            Icons.insights_rounded,
            _amber,
            _i,
            (v) => setState(() {
                  _i = v;
                  widget.onChanged(_p, _f, _m, _i);
                })),
        const SizedBox(height: 8),
        GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16),
                decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                    borderRadius: BorderRadius.circular(18),
                    boxShadow: [
                      BoxShadow(
                          color: _pink.withOpacity(.35),
                          blurRadius: 14,
                          offset: const Offset(0, 5))
                    ]),
                child: const Center(
                    child: Text('Save & Close',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none))))),
      ])),
    );
  }

  Widget _upcoming(IconData icon, Color c, String t, String s, String trail) =>
      Container(
          margin: const EdgeInsets.only(bottom: 8),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFFFCE7F3), width: 1.2)),
          child: Row(children: [
            Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    color: c.withOpacity(.12),
                    borderRadius: BorderRadius.circular(9)),
                child: Icon(icon, color: c, size: 16)),
            const SizedBox(width: 10),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(t,
                      style: const TextStyle(
                          color: _textDark,
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          decoration: TextDecoration.none)),
                  Text(s,
                      style: const TextStyle(
                          color: _textMid,
                          fontSize: 11,
                          decoration: TextDecoration.none)),
                ])),
            Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                    color: c.withOpacity(.10),
                    borderRadius: BorderRadius.circular(20)),
                child: Text(trail,
                    style: TextStyle(
                        color: c,
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        decoration: TextDecoration.none))),
          ]));

  Widget _tog(String l, String s, IconData icon, Color c, bool v,
          ValueChanged<bool> fn) =>
      Container(
          margin: const EdgeInsets.only(bottom: 10),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
              color: v ? c.withOpacity(.06) : Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                  color: v ? c.withOpacity(.25) : const Color(0xFFFCE7F3),
                  width: 1.2)),
          child: Row(children: [
            Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    color: c.withOpacity(.12),
                    borderRadius: BorderRadius.circular(9)),
                child: Icon(icon, color: c, size: 16)),
            const SizedBox(width: 10),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(l,
                      style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: v ? _textDark : Colors.grey.shade500,
                          decoration: TextDecoration.none)),
                  Text(s,
                      style: const TextStyle(
                          fontSize: 11,
                          color: Color(0xFFCCBBDD),
                          decoration: TextDecoration.none)),
                ])),
            Switch(
                value: v,
                onChanged: fn,
                activeColor: c,
                activeTrackColor: c.withOpacity(.25),
                inactiveThumbColor: Colors.grey.shade300,
                inactiveTrackColor: Colors.grey.shade100,
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap),
          ]));

  String _sm(int m) => const [
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
      ][m - 1];
}

// ═══════════════════════════════════════════════════════════════════════════
// RING PAINTER
// ═══════════════════════════════════════════════════════════════════════════

class _RingPainter extends CustomPainter {
  final double progress;
  final int periodDuration;
  final int cycleLength;
  const _RingPainter(
      {required this.progress,
      required this.periodDuration,
      required this.cycleLength});

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2 - 14;
    const w = 14.0;
    const full = 2 * math.pi;
    const start = -math.pi / 2;

    // Background track
    canvas.drawCircle(
        c,
        r,
        Paint()
          ..color = const Color(0xFFF5E6F0)
          ..strokeWidth = w
          ..style = PaintingStyle.stroke);

    void arc(double s, double sw, Color color) => canvas.drawArc(
        Rect.fromCircle(center: c, radius: r),
        s,
        sw,
        false,
        Paint()
          ..color = color
          ..strokeWidth = w
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round);

    final pf = periodDuration / cycleLength;
    final fe = (cycleLength - 14 - 5) / cycleLength;
    final oe = (cycleLength - 14 + 2) / cycleLength;

    arc(start, full * pf, _pink);
    arc(start + full * pf, full * (fe - pf), _teal);
    arc(start + full * fe, full * (oe - fe), _purple);
    arc(start + full * oe, full * (1 - oe), _amber);

    // Progress dot
    final da = start + full * progress;
    final dx = c.dx + r * math.cos(da);
    final dy = c.dy + r * math.sin(da);
    canvas.drawCircle(
        Offset(dx, dy),
        10,
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.fill);
    canvas.drawCircle(
        Offset(dx, dy),
        7,
        Paint()
          ..color = _pink
          ..style = PaintingStyle.fill);
  }

  @override
  bool shouldRepaint(_RingPainter o) => o.progress != progress;
}

// ═══════════════════════════════════════════════════════════════════════════
// SMALL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

Widget _handle() => Center(
    child: Container(
        margin: const EdgeInsets.only(top: 12, bottom: 20),
        width: 44,
        height: 5,
        decoration: BoxDecoration(
            color: const Color(0xFFE0C8D8),
            borderRadius: BorderRadius.circular(3))));

class _Lbl extends StatelessWidget {
  final String text;
  const _Lbl(this.text);
  @override
  Widget build(BuildContext context) => Text(text,
      style: const TextStyle(
          color: _textDark,
          fontSize: 13,
          fontWeight: FontWeight.w700,
          decoration: TextDecoration.none));
}

class _StepperWidget extends StatelessWidget {
  final int value, min, max;
  final ValueChanged<int> onChanged;
  const _StepperWidget(
      {required this.value,
      required this.min,
      required this.max,
      required this.onChanged});

  @override
  Widget build(BuildContext context) => Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        GestureDetector(
            onTap: value > min ? () => onChanged(value - 1) : null,
            child: Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                    color: value > min
                        ? const Color(0xFFFFF0F7)
                        : Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(10)),
                child: Icon(Icons.remove_rounded,
                    color: value > min ? _pink : Colors.grey.shade300,
                    size: 18))),
        Text('$value',
            style: const TextStyle(
                color: _textDark,
                fontSize: 20,
                fontWeight: FontWeight.w900,
                decoration: TextDecoration.none)),
        GestureDetector(
            onTap: value < max ? () => onChanged(value + 1) : null,
            child: Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                    color: value < max
                        ? const Color(0xFFFFF0F7)
                        : Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(10)),
                child: Icon(Icons.add_rounded,
                    color: value < max ? _pink : Colors.grey.shade300,
                    size: 18))),
      ]));
}

class _Loader extends StatelessWidget {
  const _Loader();
  @override
  Widget build(BuildContext context) => const Center(
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        CircularProgressIndicator(
            valueColor: AlwaysStoppedAnimation(_pink), strokeWidth: 3),
        SizedBox(height: 16),
        Text('Loading cycle data…',
            style: TextStyle(
                color: _textMid,
                fontSize: 14,
                decoration: TextDecoration.none)),
      ]));
}
