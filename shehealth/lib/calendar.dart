// calendar.dart – She Health · Period Cycle Monitoring

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

const String kApiBase    = 'http://localhost:8000';
const String kHistoryKey = 'cycle_history_local';
const String kMLPredKey  = 'ml_prediction_local';
const String kPeriodKey          = 'period_days_v1';
const String kPeriodDataPrefsKey  = 'period_days_v1'; // alias for dashboard

const _pink      = Color(0xFFC85A7A);
const _pinkLight = Color(0xFFE87DAB);
const _purple    = Color(0xFF9B84D4);
const _teal      = Color(0xFF6DBFB0);
const _amber     = Color(0xFFE8A838);
const _dark      = Color(0xFF2D1B2E);
const _mid       = Color(0xFFBBAACC);

// ─────────────────────────── MODELS ──────────────────────────────────────────

class CycleEntry {
  final String?  id;
  final DateTime startDate;
  final int      cycleLength;
  final int      periodDuration;
  final bool     isHistorical;
  final bool     unusualBleeding;

  const CycleEntry({
    this.id,
    required this.startDate,
    required this.cycleLength,
    this.periodDuration  = 5,
    this.isHistorical    = false,
    this.unusualBleeding = false,
  });

  DateTime get ovulationDay => startDate.add(Duration(days: cycleLength - 14));
  DateTime get nextPeriod   => startDate.add(Duration(days: cycleLength));

  List<DateTime> get periodDays  =>
      List.generate(periodDuration, (i) => startDate.add(Duration(days: i)));
  List<DateTime> get fertileDays =>
      List.generate(6, (i) => ovulationDay.subtract(Duration(days: 5 - i)));
  List<DateTime> get pmsDays =>
      List.generate(5, (i) => nextPeriod.subtract(Duration(days: 5 - i)));

  int get currentDay    => (DateTime.now().difference(startDate).inDays + 1).clamp(1, cycleLength);
  int get daysUntilNext => nextPeriod.difference(DateTime.now()).inDays;

  String get phase {
    final d = currentDay;
    if (d <= periodDuration)           return 'Menstrual';
    if (d <= cycleLength - 14 - 5)     return 'Follicular';
    if (d <= cycleLength - 14 + 1)     return 'Ovulatory';
    return 'Luteal';
  }

  String get phaseEmoji {
    switch (phase) {
      case 'Menstrual':  return '🩸';
      case 'Follicular': return '🌱';
      case 'Ovulatory':  return '🌸';
      default:           return '🌙';
    }
  }

  Color get phaseColor {
    switch (phase) {
      case 'Menstrual':  return _pink;
      case 'Follicular': return _teal;
      case 'Ovulatory':  return _purple;
      default:           return _amber;
    }
  }

  String get phaseTip {
    switch (phase) {
      case 'Menstrual':  return 'Rest and stay hydrated. Use heat pads for cramps.';
      case 'Follicular': return 'Energy rising! Great time for new activities.';
      case 'Ovulatory':  return 'Peak fertility window. Best time for conception.';
      default:           return 'Prioritise sleep and self-care. PMS may appear.';
    }
  }

  List<int> periodDaysForMonth(int y, int m) =>
      periodDays.where((d) => d.year == y && d.month == m).map((d) => d.day).toList();
  List<int> fertileDaysForMonth(int y, int m) =>
      fertileDays.where((d) => d.year == y && d.month == m).map((d) => d.day).toList();
  int? ovulDayForMonth(int y, int m) =>
      (ovulationDay.year == y && ovulationDay.month == m) ? ovulationDay.day : null;
  List<int> pmsDaysForMonth(int y, int m) =>
      pmsDays.where((d) => d.year == y && d.month == m).map((d) => d.day).toList();

  Map<String, dynamic> toJson() => {
    if (id != null) 'id': id,
    'cycle_start_date': startDate.toIso8601String(),
    'cycle_length':     cycleLength,
    'period_duration':  periodDuration,
    'is_historical':    isHistorical,
    'unusual_bleeding': unusualBleeding,
  };

  factory CycleEntry.fromJson(Map<String, dynamic> j) => CycleEntry(
    id:              j['id'] as String?,
    startDate:       DateTime.parse(j['cycle_start_date'] as String),
    cycleLength:     (j['cycle_length'] as num).toInt(),
    periodDuration:  (j['period_duration'] as num?)?.toInt() ?? 5,
    isHistorical:    (j['is_historical'] as bool?) ?? false,
    unusualBleeding: (j['unusual_bleeding'] as bool?) ?? false,
  );
}

class MLPrediction {
  final int    ovulationDay;
  final String ovulationDate, fertileStart, fertileEnd;
  final String nextPeriod, pmsStart;
  final int    lutealLength;
  final String source, confidence;

  const MLPrediction({
    required this.ovulationDay, required this.ovulationDate,
    required this.fertileStart, required this.fertileEnd,
    required this.nextPeriod,   required this.pmsStart,
    required this.lutealLength, required this.source,
    required this.confidence,
  });

  factory MLPrediction.fromJson(Map<String, dynamic> j) => MLPrediction(
    ovulationDay:  (j['predicted_ovulation_day'] as num).toInt(),
    ovulationDate: j['ovulation_date']       as String,
    fertileStart:  j['fertile_window_start']  as String,
    fertileEnd:    j['fertile_window_end']    as String,
    nextPeriod:    j['next_period_date']      as String,
    pmsStart:      j['pms_window_start']      as String,
    lutealLength:  (j['luteal_phase_length']  as num).toInt(),
    source:        j['prediction_source']     as String,
    confidence:    j['confidence']            as String,
  );

  Color get confColor {
    switch (confidence) {
      case 'high':   return _teal;
      case 'medium': return _amber;
      default:       return _pink;
    }
  }
  bool get isML => source == 'ml_model';
}

// ─────────────────────────── DAILY LOG MODEL ─────────────────────────────────

class DailyLog {
  final String   date;      // yyyy-MM-dd
  final String   mood;      // happy | sad | anxious | irritable | calm
  final String   flow;      // none | light | medium | heavy
  final List<String> symptoms; // cramps | headache | bloating | fatigue
  const DailyLog({
    required this.date,
    this.mood      = '',
    this.flow      = 'none',
    this.symptoms  = const [],
  });
  Map<String, dynamic> toJson() => {
    'date': date, 'mood': mood, 'flow': flow, 'symptoms': symptoms,
  };
  factory DailyLog.fromJson(Map<String, dynamic> j) => DailyLog(
    date:     j['date'] as String,
    mood:     (j['mood'] as String?) ?? '',
    flow:     (j['flow'] as String?) ?? 'none',
    symptoms: List<String>.from((j['symptoms'] as List?) ?? []),
  );
}


// ─────────────────────────── API ─────────────────────────────────────────────

class _Api {
  static Future<bool> logCycle(CycleEntry e, String uid) async {
    try {
      final r = await http.post(
        Uri.parse('$kApiBase/api/cycle/log'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': uid, ...e.toJson()}),
      ).timeout(const Duration(seconds: 10));
      return r.statusCode == 200 || r.statusCode == 201;
    } catch (_) { return false; }
  }

  static Future<bool> deleteCycle(String id, String uid) async {
    try {
      final r = await http.delete(
        Uri.parse('$kApiBase/api/cycle/$id?user_id=$uid'),
      ).timeout(const Duration(seconds: 10));
      return r.statusCode == 204 || r.statusCode == 200;
    } catch (_) { return false; }
  }

  static Future<List<CycleEntry>> fetchHistory(String uid) async {
    try {
      final r = await http.get(
        Uri.parse('$kApiBase/api/cycle/history?user_id=$uid&limit=24'),
      ).timeout(const Duration(seconds: 10));
      if (r.statusCode != 200) return [];
      return (jsonDecode(r.body) as List)
          .map((e) => CycleEntry.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) { return []; }
  }

  static Future<MLPrediction?> predict({
    required String uid,
    required String startDate,
    required int cycleLength,
    required int cycleNumber,
    required int unusualBleeding,
    required int periodDuration,
  }) async {
    try {
      final r = await http.post(
        Uri.parse('$kApiBase/api/cycle-predict/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id':          uid,
          'cycle_start_date': startDate,
          'length_of_cycle':  cycleLength,
          'cycle_number':     cycleNumber,
          'unusual_bleeding': unusualBleeding,
          'phases_bleeding':  periodDuration,
          'length_of_luteal': cycleLength - 14,
        }),
      ).timeout(const Duration(seconds: 15));
      if (r.statusCode != 200) return null;
      return MLPrediction.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
    } catch (_) { return null; }
  }

  static Future<MLPrediction?> bulkPredict({
    required String uid,
    required List<CycleEntry> history,
  }) async {
    try {
      final sorted = [...history]..sort((a, b) => a.startDate.compareTo(b.startDate));
      final cycles = sorted.map((e) => {
        'user_id':          uid,
        'cycle_start_date': e.startDate.toIso8601String().split('T')[0],
        'length_of_cycle':  e.cycleLength,
        'unusual_bleeding': e.unusualBleeding ? 1 : 0,
        'phases_bleeding':  e.periodDuration,
        'length_of_luteal': e.cycleLength - 14,
      }).toList();
      final r = await http.post(
        Uri.parse('$kApiBase/api/cycle-predict/bulk'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': uid, 'cycles': cycles}),
      ).timeout(const Duration(seconds: 20));
      if (r.statusCode != 200) return null;
      final list = jsonDecode(r.body) as List;
      if (list.isEmpty) return null;
      return MLPrediction.fromJson(list.last as Map<String, dynamic>);
    } catch (_) { return null; }
  }

  static Future<bool> saveDailyLog(DailyLog log, String uid) async {
    try {
      final r = await http.post(
        Uri.parse('$kApiBase/api/cycle/daily-log'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': uid, ...log.toJson(),
          'flow': _flowToInt(log.flow), 'moods': [log.mood],
          'symptoms': log.symptoms, 'note': ''}),
      ).timeout(const Duration(seconds: 10));
      return r.statusCode == 200 || r.statusCode == 201;
    } catch (_) { return false; }
  }

  static int _flowToInt(String f) {
    switch (f) {
      case 'light':  return 1;
      case 'medium': return 2;
      case 'heavy':  return 3;
      default:       return 0;
    }
  }
}

// ─────────────────────────── CACHE ───────────────────────────────────────────

class _Cache {
  static Future<List<CycleEntry>> loadHistory() async {
    try {
      final p = await SharedPreferences.getInstance();
      final raw = p.getString(kHistoryKey);
      if (raw == null) return [];
      return (jsonDecode(raw) as List)
          .map((e) => CycleEntry.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) { return []; }
  }

  static Future<void> saveHistory(List<CycleEntry> h) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(kHistoryKey, jsonEncode(h.map((e) => e.toJson()).toList()));
  }

  static Future<MLPrediction?> loadPred() async {
    try {
      final p = await SharedPreferences.getInstance();
      final raw = p.getString(kMLPredKey);
      if (raw == null) return null;
      return MLPrediction.fromJson(jsonDecode(raw) as Map<String, dynamic>);
    } catch (_) { return null; }
  }

  static Future<void> savePred(MLPrediction pred) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(kMLPredKey, jsonEncode({
      'predicted_ovulation_day': pred.ovulationDay,
      'ovulation_date':          pred.ovulationDate,
      'fertile_window_start':    pred.fertileStart,
      'fertile_window_end':      pred.fertileEnd,
      'next_period_date':        pred.nextPeriod,
      'pms_window_start':        pred.pmsStart,
      'luteal_phase_length':     pred.lutealLength,
      'prediction_source':       pred.source,
      'confidence':              pred.confidence,
    }));
  }

  // Daily logs stored as map: date → DailyLog json
  static const _kDailyLogs = 'daily_logs_local';

  static Future<Map<String, DailyLog>> loadDailyLogs() async {
    try {
      final p = await SharedPreferences.getInstance();
      final raw = p.getString(_kDailyLogs);
      if (raw == null) return {};
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map((k, v) => MapEntry(k, DailyLog.fromJson(v as Map<String, dynamic>)));
    } catch (_) { return {}; }
  }

  static Future<void> saveDailyLog(DailyLog log) async {
    final logs = await loadDailyLogs();
    logs[log.date] = log;
    final p = await SharedPreferences.getInstance();
    await p.setString(_kDailyLogs, jsonEncode(
        logs.map((k, v) => MapEntry(k, v.toJson()))));
  }

  static Future<DailyLog?> getDailyLog(String date) async {
    final logs = await loadDailyLogs();
    return logs[date];
  }
}

// ─────────────────────────── MAIN WIDGET ─────────────────────────────────────

class PeriodCalendarWidget extends StatefulWidget {
  const PeriodCalendarWidget({Key? key}) : super(key: key);
  @override
  State<PeriodCalendarWidget> createState() => _CalState();
}

class _CalState extends State<PeriodCalendarWidget> with TickerProviderStateMixin {
  List<CycleEntry>        _history   = [];
  MLPrediction?           _mlPred;
  Map<String, DailyLog>   _dailyLogs = {};
  String                  _userId    = '1';
  bool                    _loading   = true;
  bool                    _saving    = false;
  bool                    _hasUnread = true;

  DateTime _selDate = DateTime.now();
  int?     _selDay;
  int      _tab    = 0;

  // Phase tips carousel controller
  late PageController _tipCtrl;
  int _tipPage = 0;

  bool _nPeriod = true, _nFertile = true, _nMed = false, _nInsights = true;

  late AnimationController _pulseCtrl;
  late Animation<double>   _pulseAnim;

  // ── helpers ──────────────────────────────────────────────────────────────
  CycleEntry? get _latest {
    if (_history.isEmpty) return null;
    return ([..._history]..sort((a, b) => b.startDate.compareTo(a.startDate))).first;
  }

  List<int> get _perMonth {
    final s = <int>{};
    for (final e in _history) s.addAll(e.periodDaysForMonth(_selDate.year, _selDate.month));
    return s.toList()..sort();
  }

  List<int> get _ferMonth {
    final s = <int>{};
    for (final e in _history) s.addAll(e.fertileDaysForMonth(_selDate.year, _selDate.month));
    return s.toList()..sort();
  }

  List<int> get _pmsMonth {
    final s = <int>{};
    for (final e in _history) s.addAll(e.pmsDaysForMonth(_selDate.year, _selDate.month));
    return s.toList()..sort();
  }

  int? get _ovulMonth {
    for (final e in _history) {
      final d = e.ovulDayForMonth(_selDate.year, _selDate.month);
      if (d != null) return d;
    }
    return null;
  }

  int get _daysInMonth => DateTime(_selDate.year, _selDate.month + 1, 0).day;

  // ── lifecycle ─────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: .95, end: 1.05).animate(
        CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));
    _tipCtrl = PageController(viewportFraction: 0.88);
    _init();
  }

  Future<void> _init() async {
    final prefs = await SharedPreferences.getInstance();
    _userId = prefs.getString('user_id') ?? '1';
    final local = await _Cache.loadHistory();
    final cachedPred = await _Cache.loadPred();
    final cachedLogs = await _Cache.loadDailyLogs();
    if (mounted) setState(() {
      _history = local; _mlPred = cachedPred;
      _dailyLogs = cachedLogs; _loading = false;
    });

    final remote = await _Api.fetchHistory(_userId);
    if (remote.isNotEmpty && mounted) {
      await _Cache.saveHistory(remote);
      setState(() => _history = remote);
      final latest = ([...remote]..sort((a, b) => b.startDate.compareTo(a.startDate))).first;
      final pred = await _Api.predict(
        uid: _userId,
        startDate: latest.startDate.toIso8601String().split('T')[0],
        cycleLength: latest.cycleLength,
        cycleNumber: remote.length,
        unusualBleeding: latest.unusualBleeding ? 1 : 0,
        periodDuration: latest.periodDuration,
      );
      if (pred != null && mounted) {
        await _Cache.savePred(pred);
        setState(() => _mlPred = pred);
      }
    }
  }

  @override
  void dispose() { _pulseCtrl.dispose(); _tipCtrl.dispose(); super.dispose(); }

  // ── save / delete ─────────────────────────────────────────────────────────
  Future<void> _saveCycle(CycleEntry entry) async {
    setState(() => _saving = true);
    _history.removeWhere((e) => _sameDay(e.startDate, entry.startDate));
    _history.add(entry);
    _history.sort((a, b) => b.startDate.compareTo(a.startDate));
    await _Cache.saveHistory(_history);

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(kPeriodKey, jsonEncode({
      'year': entry.startDate.year, 'month': entry.startDate.month,
      'cycle_start_date': entry.startDate.toIso8601String(),
      'cycle_length': entry.cycleLength, 'period_duration': entry.periodDuration,
    }));

    final ok = await _Api.logCycle(entry, _userId);
    MLPrediction? pred;
    // Always use bulk so the model sees the full cycle history pattern.
    // With 1 cycle it behaves like single predict; with 3+ cycles the
    // model uses the variability across all cycles for better accuracy.
    if (_history.length >= 2) {
      pred = await _Api.bulkPredict(uid: _userId, history: _history);
    } else {
      // Only 1 cycle logged — use single predict
      pred = await _Api.predict(
        uid: _userId,
        startDate: entry.startDate.toIso8601String().split('T')[0],
        cycleLength: entry.cycleLength,
        cycleNumber: 1,
        unusualBleeding: entry.unusualBleeding ? 1 : 0,
        periodDuration: entry.periodDuration,
      );
    }
    if (pred != null) await _Cache.savePred(pred);
    if (mounted) {
      setState(() { _saving = false; if (pred != null) _mlPred = pred; });
      _snack(ok ? '✅ Cycle saved & predicted!' : '💾 Saved locally (offline)');
    }
  }

  Future<void> _saveSymptoms(DailyLog log) async {
    await _Cache.saveDailyLog(log);
    setState(() => _dailyLogs[log.date] = log);
    await _Api.saveDailyLog(log, _userId);
  }

  Future<void> _deleteCycle(CycleEntry e) async {
    final ok = await _confirm('Delete Cycle', 'Remove cycle starting ${_fmt(e.startDate)}?');
    if (!ok) return;
    setState(() => _history.removeWhere((h) => _sameDay(h.startDate, e.startDate)));
    await _Cache.saveHistory(_history);
    if (e.id != null) await _Api.deleteCycle(e.id!, _userId);
    // Re-predict from remaining history so ML card stays current
    if (_history.isNotEmpty) {
      final latest = ([..._history]..sort((a, b) => b.startDate.compareTo(a.startDate))).first;
      final pred = await _Api.predict(
        uid: _userId,
        startDate: latest.startDate.toIso8601String().split('T')[0],
        cycleLength: latest.cycleLength,
        cycleNumber: _history.length,
        unusualBleeding: latest.unusualBleeding ? 1 : 0,
        periodDuration: latest.periodDuration,
      );
      if (pred != null && mounted) {
        await _Cache.savePred(pred);
        setState(() => _mlPred = pred);
      }
    } else {
      setState(() => _mlPred = null);
    }
    _snack('Cycle removed');
  }

  // ── build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFFF0F7),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [Color(0xFFFFF0F7), Color(0xFFFDE8F5), Color(0xFFF5E6FF), Colors.white],
            stops: [0, .3, .6, 1],
          ),
        ),
        child: _loading
            ? const _Loader()
            : SafeArea(
                child: SingleChildScrollView(
                  physics: const ClampingScrollPhysics(),
                  child: Column(
                    children: [_header(), _ring(), _navRow(), _tabsAndContent()],
                  ),
                ),
              ),
      ),
    );
  }

  // ── header ────────────────────────────────────────────────────────────────
  Widget _header() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(_mn(_selDate.month), style: const TextStyle(
                color: _pink, fontSize: 22, fontWeight: FontWeight.w800,
                decoration: TextDecoration.none)),
            Text('${_selDate.year}', style: const TextStyle(
                color: Color(0xFFD4A0B8), fontSize: 13, fontWeight: FontWeight.w500,
                decoration: TextDecoration.none)),
          ]),
          Row(children: [
            Stack(clipBehavior: Clip.none, children: [
              _iconBtn(Icons.notifications_none_rounded, onTap: _showNotifPanel),
              if (_hasUnread)
                Positioned(top: -2, right: -2,
                  child: Container(width: 10, height: 10,
                    decoration: BoxDecoration(color: _pink, shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 1.5)))),
            ]),
            const SizedBox(width: 10),
            _iconBtn(Icons.calendar_month_rounded, highlighted: true, onTap: _showCalModal),
          ]),
        ],
      ),
    );
  }

  Widget _iconBtn(IconData icon, {bool highlighted = false, VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44, height: 44,
        decoration: BoxDecoration(
          color: highlighted ? _pink : Colors.white,
          borderRadius: BorderRadius.circular(14),
          boxShadow: [BoxShadow(
            color: highlighted ? _pink.withOpacity(.35) : Colors.black.withOpacity(.08),
            blurRadius: highlighted ? 12 : 8, offset: const Offset(0, 3))]),
        child: Icon(icon, color: highlighted ? Colors.white : _pink, size: 22),
      ),
    );
  }

  // ── ring ──────────────────────────────────────────────────────────────────
  Widget _ring() {
    final total  = _daysInMonth;
    const radius = 108.0;
    final perSet = _perMonth.toSet();
    final ferSet = _ferMonth.toSet();
    final ovul   = _ovulMonth;
    final pmsSet = _pmsMonth.toSet();

    return SizedBox(
      height: 256,
      child: Stack(alignment: Alignment.center, children: [
        Container(width: 128, height: 128,
          decoration: BoxDecoration(shape: BoxShape.circle,
            gradient: RadialGradient(colors: [
              const Color(0xFFFFD6E8).withOpacity(.6), Colors.transparent]))),
        ...List.generate(total, (i) {
          final day   = i + 1;
          final angle = (2 * math.pi / total) * i - math.pi / 2;
          final isPer  = perSet.contains(day);
          final isFer  = ferSet.contains(day);
          final isOvul = ovul == day;
          final isPms  = pmsSet.contains(day);
          final isSel  = _selDay == day;
          final isTod  = day == DateTime.now().day &&
              _selDate.month == DateTime.now().month && _selDate.year == DateTime.now().year;
          return Transform.translate(
            offset: Offset(radius * math.cos(angle), radius * math.sin(angle)),
            child: GestureDetector(
              onTap: () => setState(() => _selDay = day),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: isSel ? 34 : 28, height: isSel ? 34 : 28,
                decoration: BoxDecoration(
                  gradient: isPer
                      ? const LinearGradient(colors: [_pinkLight, _pink],
                          begin: Alignment.topLeft, end: Alignment.bottomRight)
                      : isOvul
                          ? const LinearGradient(colors: [Color(0xFF80D8CC), _teal],
                              begin: Alignment.topLeft, end: Alignment.bottomRight)
                          : isFer
                              ? const LinearGradient(colors: [Color(0xFFB5A4E0), _purple],
                                  begin: Alignment.topLeft, end: Alignment.bottomRight)
                              : null,
                  color: (!isPer && !isFer && !isOvul)
                      ? isPms ? const Color(0xFFFFF3CD)
                      : isTod ? const Color(0xFFFFD6E8)
                      : isSel ? const Color(0xFFEED8F0)
                      : const Color(0xFFF5E6F5).withOpacity(.6) : null,
                  shape: BoxShape.circle,
                  border: isSel && !isPer
                      ? Border.all(color: _pink, width: 2.5)
                      : isTod && !isPer
                          ? Border.all(color: _pinkLight.withOpacity(.5), width: 1.5) : null,
                  boxShadow: isPer
                      ? [BoxShadow(color: _pink.withOpacity(.35), blurRadius: 6, offset: const Offset(0, 2))]
                      : (isFer || isOvul)
                          ? [BoxShadow(color: (isOvul ? _teal : _purple).withOpacity(.35), blurRadius: 6, offset: const Offset(0, 2))]
                          : null,
                ),
                child: Center(child: Text('$day', style: TextStyle(
                  color: isPer || isFer || isOvul ? Colors.white
                      : isTod ? _pink : isPms ? _amber : const Color(0xFFCCA8C0),
                  fontSize: 10,
                  fontWeight: isPer || isFer || isOvul || isTod ? FontWeight.w700 : FontWeight.w500,
                  decoration: TextDecoration.none))),
              ),
            ),
          );
        }),
        _centre(),
      ]),
    );
  }

  Widget _centre() {
    final day = _selDay ?? DateTime.now().day;
    return Column(mainAxisSize: MainAxisSize.min, children: [
      Text(_dayName(DateTime(_selDate.year, _selDate.month, day.clamp(1, _daysInMonth))),
          style: const TextStyle(color: Color(0xFFBB8FAE), fontSize: 11,
              fontWeight: FontWeight.w600, letterSpacing: 2, decoration: TextDecoration.none)),
      const SizedBox(height: 4),
      Row(mainAxisSize: MainAxisSize.min, children: [
        _navArrow(Icons.chevron_left_rounded, () => setState(() {
          _selDate = DateTime(_selDate.year, _selDate.month - 1, 1); _selDay = null;
        })),
        const SizedBox(width: 6),
        Column(children: [
          ScaleTransition(scale: _pulseAnim,
            child: Text('$day', style: const TextStyle(
                color: _pink, fontSize: 44, fontWeight: FontWeight.w900,
                height: 1, letterSpacing: -2, decoration: TextDecoration.none))),
          Text('${_sn(_selDate.month)} ${_selDate.year}',
              style: const TextStyle(color: Color(0xFFE087A8), fontSize: 13,
                  fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
        ]),
        const SizedBox(width: 6),
        _navArrow(Icons.chevron_right_rounded, () => setState(() {
          _selDate = DateTime(_selDate.year, _selDate.month + 1, 1); _selDay = null;
        })),
      ]),
      const SizedBox(height: 8),
      Row(mainAxisSize: MainAxisSize.min, children: [
        _badge('${_perMonth.length}d period', _pink),
        const SizedBox(width: 8),
        _badge('day $day', _purple),
      ]),
      if (_saving) ...[
        const SizedBox(height: 6),
        const SizedBox(width: 14, height: 14,
            child: CircularProgressIndicator(strokeWidth: 2, color: _pink)),
      ],
    ]);
  }

  Widget _navArrow(IconData icon, VoidCallback fn) => GestureDetector(
    onTap: fn,
    child: Container(width: 30, height: 30,
      decoration: BoxDecoration(color: const Color(0xFFFFF0F7), shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFFEEC4D6), width: 1)),
      child: Icon(icon, color: _pink, size: 18)));

  Widget _badge(String text, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(color: color.withOpacity(.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(.25), width: 1)),
    child: Text(text, style: TextStyle(color: color, fontSize: 11,
        fontWeight: FontWeight.w600, decoration: TextDecoration.none)));

  // ── nav row ───────────────────────────────────────────────────────────────
  Widget _navRow() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 2),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        _yearBtn(Icons.keyboard_double_arrow_left_rounded, () => setState(() {
          _selDate = DateTime(_selDate.year - 1, _selDate.month); _selDay = null;
        })),
        Row(children: [
          _ldot(_pink, 'Period'), const SizedBox(width: 12), _ldot(_purple, 'Fertile'),
        ]),
        _yearBtn(Icons.keyboard_double_arrow_right_rounded, () => setState(() {
          _selDate = DateTime(_selDate.year + 1, _selDate.month); _selDay = null;
        })),
      ]),
    );
  }

  Widget _yearBtn(IconData icon, VoidCallback fn) => GestureDetector(
    onTap: fn,
    child: Container(padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(.06), blurRadius: 6, offset: const Offset(0, 2))]),
      child: Icon(icon, color: _pink, size: 20)));

  Widget _ldot(Color c, String l) => Row(children: [
    Container(width: 10, height: 10, decoration: BoxDecoration(color: c, shape: BoxShape.circle,
        boxShadow: [BoxShadow(color: c.withOpacity(.4), blurRadius: 4)])),
    const SizedBox(width: 5),
    Text(l, style: TextStyle(color: c, fontSize: 11, fontWeight: FontWeight.w600,
        decoration: TextDecoration.none))]);

  // ── tabs + content ────────────────────────────────────────────────────────
  Widget _tabsAndContent() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Container(
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(color: const Color(0xFFF5E6F5).withOpacity(.7),
              borderRadius: BorderRadius.circular(16)),
          child: Row(children: [
            _tabBtn(0, Icons.edit_note_rounded, 'Log Day'),
            _tabBtn(1, Icons.history_rounded, 'History'),
            _tabBtn(2, Icons.insights_rounded, 'Insights'),
          ])),
        const SizedBox(height: 12),
        Container(
          width: double.infinity,
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(24),
              boxShadow: [BoxShadow(color: _pink.withOpacity(.06), blurRadius: 20, offset: const Offset(0, 4))]),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: _tab == 0 ? _logDayTab() : _tab == 1 ? _historyTab() : _insightsTab())),
      ]));
  }

  Widget _tabBtn(int idx, IconData icon, String label) {
    final sel = _tab == idx;
    return Expanded(child: GestureDetector(
      onTap: () => setState(() => _tab = idx),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        padding: const EdgeInsets.symmetric(vertical: 11),
        decoration: BoxDecoration(
          color: sel ? Colors.white : Colors.transparent,
          borderRadius: BorderRadius.circular(13),
          boxShadow: sel ? [BoxShadow(color: Colors.black.withOpacity(.08), blurRadius: 8, offset: const Offset(0, 2))] : null),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(icon, color: sel ? _pink : _mid, size: 16),
          const SizedBox(width: 5),
          Text(label, style: TextStyle(color: sel ? _pink : _mid, fontSize: 13,
              fontWeight: sel ? FontWeight.w700 : FontWeight.w500, decoration: TextDecoration.none)),
        ]))));
  }

  // ── log day tab ───────────────────────────────────────────────────────────
  Widget _logDayTab() {
    final cd = _latest;
    if (cd == null) return _startPrompt();
    final day    = _selDay ?? DateTime.now().day;
    final isPer  = _perMonth.contains(day);
    final isFer  = _ferMonth.contains(day);
    final dateStr = '${_selDate.year}-${_selDate.month.toString().padLeft(2,'0')}-${day.toString().padLeft(2,'0')}';
    final todayLog = _dailyLogs[dateStr];

    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      // Header with edit button
      Row(children: [
        Container(padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(color: const Color(0xFFFFEEF5), borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.water_drop_rounded, color: _pink, size: 18)),
        const SizedBox(width: 10),
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('${_mn(_selDate.month)} $day', style: const TextStyle(color: _dark, fontSize: 16,
              fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
          Text(isPer ? 'Period day ✓' : isFer ? 'Fertile window 💜' : 'No period today',
              style: TextStyle(color: isPer ? _pink : isFer ? _purple : _mid,
                  fontSize: 12, fontWeight: FontWeight.w500, decoration: TextDecoration.none)),
        ]),
        const Spacer(),
        GestureDetector(
          onTap: () => _showLogSheet(edit: cd),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_pinkLight, _pink]),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [BoxShadow(color: _pink.withOpacity(.3), blurRadius: 8, offset: const Offset(0, 3))]),
            child: const Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.edit_rounded, color: Colors.white, size: 14),
              SizedBox(width: 5),
              Text('Edit Cycle', style: TextStyle(color: Colors.white, fontSize: 12,
                  fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
            ]))),
      ]),
      const SizedBox(height: 14),

      // ── Phase tips carousel ──────────────────────────────────────────────
      _phaseTipsCarousel(cd),
      const SizedBox(height: 14),

      // ── Log Symptoms button ───────────────────────────────────────────────
      GestureDetector(
        onTap: () => _showSymptomSheet(dateStr, todayLog),
        child: Container(
          width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: todayLog != null ? _purple.withOpacity(.08) : Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: todayLog != null ? _purple.withOpacity(.3) : const Color(0xFFE5D4F0),
              width: 1.5)),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(todayLog != null ? Icons.check_circle_rounded : Icons.add_circle_outline_rounded,
                color: _purple, size: 20),
            const SizedBox(width: 10),
            Text(todayLog != null ? 'Symptoms logged ✓  Tap to edit' : 'Log Today\'s Symptoms',
                style: TextStyle(color: _purple, fontSize: 14,
                    fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
          ]))),

      // Show logged symptoms summary
      if (todayLog != null) ...[
        const SizedBox(height: 10),
        _symptomSummary(todayLog),
      ],
      const SizedBox(height: 14),

      // ML card
      if (_mlPred != null) ...[_mlCard(), const SizedBox(height: 14)],

      // Stat cards
      Row(children: [
        Expanded(child: _statCard('${_perMonth.length}', 'Period days', _pink, Icons.calendar_today_rounded)),
        const SizedBox(width: 12),
        Expanded(child: _statCard('${_ferMonth.length}', 'Fertile days', _purple, Icons.favorite_rounded)),
      ]),
      const SizedBox(height: 12),
      Row(children: [
        Expanded(child: _statCard(cd.daysUntilNext > 0 ? '${cd.daysUntilNext}d' : 'Due!',
            'Next period', _teal, Icons.water_drop_rounded)),
        const SizedBox(width: 12),
        Expanded(child: _statCard('Day ${cd.currentDay}', 'Cycle day', _amber, Icons.loop_rounded)),
      ]),
    ]);
  }

  // ── Phase tips carousel ─────────────────────────────────────────────────────
  Widget _phaseTipsCarousel(CycleEntry cd) {
    final tips = _phaseTips(cd.phase);
    return Column(children: [
      SizedBox(
        height: 110,
        child: PageView.builder(
          controller: _tipCtrl,
          itemCount: tips.length,
          onPageChanged: (i) => setState(() => _tipPage = i),
          itemBuilder: (_, i) => Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: cd.phaseColor.withOpacity(.07),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: cd.phaseColor.withOpacity(.2))),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(tips[i]['icon']!, style: const TextStyle(fontSize: 22, decoration: TextDecoration.none)),
                const SizedBox(width: 10),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(tips[i]['title']!, style: TextStyle(color: cd.phaseColor, fontSize: 12,
                      fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
                  const SizedBox(height: 4),
                  Text(tips[i]['body']!, style: TextStyle(color: _dark.withOpacity(.65),
                      fontSize: 11, height: 1.4, decoration: TextDecoration.none)),
                ])),
              ]))),
        )),
      const SizedBox(height: 8),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: List.generate(tips.length, (i) =>
        AnimatedContainer(duration: const Duration(milliseconds: 250),
          margin: const EdgeInsets.symmetric(horizontal: 3),
          width: _tipPage == i ? 18 : 6, height: 6,
          decoration: BoxDecoration(
            color: _tipPage == i ? cd.phaseColor : cd.phaseColor.withOpacity(.3),
            borderRadius: BorderRadius.circular(3))))),
    ]);
  }

  List<Map<String, String>> _phaseTips(String phase) {
    switch (phase) {
      case 'Menstrual': return [
        {'icon':'🥗','title':'Nutrition','body':'Eat iron-rich foods like leafy greens, lentils and dark chocolate. Stay hydrated and avoid caffeine.'},
        {'icon':'🧘','title':'Exercise','body':'Light yoga, stretching or gentle walks. Avoid intense workouts. Heat pads help with cramps.'},
        {'icon':'😌','title':'Mood','body':'Be gentle with yourself. Rest is productive. Journalling or a warm bath can ease emotional heaviness.'},
      ];
      case 'Follicular': return [
        {'icon':'🥦','title':'Nutrition','body':'Focus on fresh vegetables and fermented foods. Estrogen rising — your gut loves probiotic support now.'},
        {'icon':'🏃','title':'Exercise','body':'Great time for cardio, strength training or trying a new workout. Energy is building — use it!'},
        {'icon':'✨','title':'Mood','body':'You\'re likely feeling optimistic and social. Great time for creative projects and making plans.'},
      ];
      case 'Ovulatory': return [
        {'icon':'🫐','title':'Nutrition','body':'Anti-inflammatory foods like berries, salmon and flaxseed support this peak phase. Light and energising meals.'},
        {'icon':'💪','title':'Exercise','body':'Peak performance phase. Ideal for HIIT, heavy lifting or competitive sports. Your body is at its strongest.'},
        {'icon':'🌸','title':'Mood','body':'You\'re at your most communicative and confident. Perfect for important conversations and presentations.'},
      ];
      default: return [ // Luteal
        {'icon':'🍫','title':'Nutrition','body':'Magnesium-rich foods like nuts, seeds and dark chocolate ease PMS. Reduce salt to minimise bloating.'},
        {'icon':'🚶','title':'Exercise','body':'Moderate exercise like walking, swimming or pilates. Intense workouts may feel harder — that\'s normal.'},
        {'icon':'🌙','title':'Mood','body':'Prioritise sleep (8h+), limit screens before bed. If mood dips sharply, track it — it\'s useful data.'},
      ];
    }
  }

  // ── Symptom summary chip row ─────────────────────────────────────────────────
  Widget _symptomSummary(DailyLog log) {
    final items = <Widget>[];
    if (log.mood.isNotEmpty) {
      final moodEmoji = {'happy':'😊','sad':'😢','anxious':'😰','irritable':'😤','calm':'😌'}[log.mood] ?? '😐';
      items.add(_summaryChip('$moodEmoji ${log.mood}', _purple));
    }
    if (log.flow != 'none' && log.flow.isNotEmpty) {
      items.add(_summaryChip('💧 ${log.flow} flow', _pink));
    }
    for (final s in log.symptoms) {
      final e = {'cramps':'😣','headache':'🤕','bloating':'😮‍💨','fatigue':'😴'}[s] ?? '•';
      items.add(_summaryChip('$e $s', _amber));
    }
    if (items.isEmpty) return const SizedBox.shrink();
    return Wrap(spacing: 6, runSpacing: 6, children: items);
  }

  Widget _summaryChip(String label, Color c) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(color: c.withOpacity(.10), borderRadius: BorderRadius.circular(20),
        border: Border.all(color: c.withOpacity(.25))),
    child: Text(label, style: TextStyle(color: c, fontSize: 11,
        fontWeight: FontWeight.w600, decoration: TextDecoration.none)));

  Widget _startPrompt() => Column(children: [
    GestureDetector(
      onTap: () => _showLogSheet(),
      child: Container(width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(gradient: const LinearGradient(colors: [_pinkLight, _pink]),
            borderRadius: BorderRadius.circular(18),
            boxShadow: [BoxShadow(color: _pink.withOpacity(.3), blurRadius: 12, offset: const Offset(0, 4))]),
        child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.add_circle_outline_rounded, color: Colors.white, size: 22),
          SizedBox(width: 10),
          Text('Start Tracking Cycle', style: TextStyle(color: Colors.white, fontSize: 16,
              fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
        ]))),
    const SizedBox(height: 10),
    GestureDetector(
      onTap: () => _showLogSheet(isHistory: true),
      child: Container(width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 13),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18),
            border: Border.all(color: _pink.withOpacity(.3), width: 1.5)),
        child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.history_rounded, color: _pink, size: 20),
          SizedBox(width: 8),
          Text('Add Past Cycle', style: TextStyle(color: _pink, fontSize: 14,
              fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
        ]))),
  ]);

  Widget _mlCard() {
    final p = _mlPred!;
    // Friendly accuracy label shown to user
    final accuracyLabel = p.confidence == 'high' ? 'High accuracy'
        : p.confidence == 'medium' ? 'Good accuracy' : 'Estimated';
    final accuracyIcon = p.confidence == 'high' ? Icons.verified_rounded
        : p.confidence == 'medium' ? Icons.check_circle_outline_rounded
        : Icons.info_outline_rounded;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF9F0FD),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _purple.withOpacity(.2), width: 1.5),
        boxShadow: [BoxShadow(color: _purple.withOpacity(.07), blurRadius: 12, offset: const Offset(0, 4))]),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        // Header
        Row(children: [
          Container(padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: _purple.withOpacity(.12), borderRadius: BorderRadius.circular(10)),
            child: const Icon(Icons.auto_awesome_rounded, color: _purple, size: 16)),
          const SizedBox(width: 10),
          const Expanded(child: Text('Upcoming Dates',
              style: TextStyle(color: _dark, fontSize: 14,
                  fontWeight: FontWeight.w800, decoration: TextDecoration.none))),
          // Accuracy pill
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: p.confColor.withOpacity(.12),
                borderRadius: BorderRadius.circular(20)),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(accuracyIcon, color: p.confColor, size: 11),
              const SizedBox(width: 4),
              Text(accuracyLabel, style: TextStyle(color: p.confColor, fontSize: 10,
                  fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
            ])),
        ]),
        const SizedBox(height: 14),
        // Date rows
        _mlRow(Icons.water_drop_rounded, _pink,   'Next Period',     p.nextPeriod),
        const SizedBox(height: 8),
        _mlRow(Icons.favorite_rounded,   _purple, 'Fertile Window',  '${p.fertileStart}  →  ${p.fertileEnd}'),
        const SizedBox(height: 8),
        _mlRow(Icons.star_rounded,       _teal,   'Ovulation',       'Around day ${p.ovulationDay}  (${p.ovulationDate})'),
        const SizedBox(height: 8),
        _mlRow(Icons.wb_cloudy_rounded,  _amber,  'PMS Window',      'From ${p.pmsStart}'),
        const SizedBox(height: 12),
        // Refresh
        GestureDetector(
          onTap: () async {
            final cd = _latest;
            if (cd == null) return;
            final np = await _Api.predict(
              uid: _userId,
              startDate: cd.startDate.toIso8601String().split('T')[0],
              cycleLength: cd.cycleLength,
              cycleNumber: _history.length,
              unusualBleeding: cd.unusualBleeding ? 1 : 0,
              periodDuration: cd.periodDuration,
            );
            if (np != null && mounted) {
              await _Cache.savePred(np);
              setState(() => _mlPred = np);
              _snack('Predictions updated!');
            }
          },
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(Icons.refresh_rounded, color: _purple.withOpacity(.6), size: 14),
            const SizedBox(width: 5),
            Text('Refresh', style: TextStyle(color: _purple.withOpacity(.6), fontSize: 11,
                fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
          ])),
      ]));
  }

  Widget _mlRow(IconData icon, Color c, String label, String value) => Row(children: [
    Container(padding: const EdgeInsets.all(6),
        decoration: BoxDecoration(color: c.withOpacity(.12), borderRadius: BorderRadius.circular(8)),
        child: Icon(icon, color: c, size: 14)),
    const SizedBox(width: 10),
    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: const TextStyle(color: _mid, fontSize: 10,
          fontWeight: FontWeight.w500, decoration: TextDecoration.none)),
      Text(value, style: const TextStyle(color: _dark, fontSize: 12,
          fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
    ])),
  ]);

  Widget _statCard(String val, String label, Color c, IconData icon) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(color: c.withOpacity(.08), borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.withOpacity(.2), width: 1)),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Icon(icon, color: c, size: 18),
      const SizedBox(height: 8),
      Text(val, style: TextStyle(color: c, fontSize: 22, fontWeight: FontWeight.w800,
          decoration: TextDecoration.none)),
      Text(label, style: TextStyle(color: c.withOpacity(.7), fontSize: 11,
          fontWeight: FontWeight.w500, decoration: TextDecoration.none)),
    ]));

  // ── history tab ───────────────────────────────────────────────────────────
  Widget _historyTab() {
    if (_history.isEmpty) {
      return Column(mainAxisSize: MainAxisSize.min, children: [
        const Text('📋', style: TextStyle(fontSize: 40, decoration: TextDecoration.none)),
        const SizedBox(height: 12),
        const Text('No history yet', style: TextStyle(color: _dark, fontSize: 16,
            fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
        const SizedBox(height: 6),
        const Text('Tap "+ Add" to log past cycles',
            style: TextStyle(color: _mid, fontSize: 12, decoration: TextDecoration.none)),
        const SizedBox(height: 20),
        GestureDetector(
          onTap: () => _showLogSheet(isHistory: true),
          child: Container(padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            decoration: BoxDecoration(gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(20)),
            child: const Text('+ Add Past Cycle', style: TextStyle(color: Colors.white,
                fontSize: 13, fontWeight: FontWeight.w700, decoration: TextDecoration.none)))),
      ]);
    }
    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        const Text('Period History', style: TextStyle(color: _dark, fontSize: 16,
            fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
        Row(children: [
          GestureDetector(
            onTap: () => _showLogSheet(isHistory: true),
            child: Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                  borderRadius: BorderRadius.circular(20)),
              child: const Text('+ Add', style: TextStyle(color: Colors.white, fontSize: 11,
                  fontWeight: FontWeight.w700, decoration: TextDecoration.none)))),
          const SizedBox(width: 8),
          Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(color: const Color(0xFFFFEEF5), borderRadius: BorderRadius.circular(20)),
            child: Text('${_history.length} cycles', style: const TextStyle(color: _pink,
                fontSize: 11, fontWeight: FontWeight.w600, decoration: TextDecoration.none))),
        ]),
      ]),
      const SizedBox(height: 14),
      ListView.separated(
        shrinkWrap: true, physics: const NeverScrollableScrollPhysics(),
        itemCount: _history.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (_, i) => _histCard(_history[i])),
    ]);
  }

  Widget _histCard(CycleEntry e) {
    final isLatest = _latest != null && _sameDay(e.startDate, _latest!.startDate);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF5F9), borderRadius: BorderRadius.circular(16),
        border: Border.all(color: isLatest ? _pink.withOpacity(.4) : const Color(0xFFEED4E0), width: 1)),
      child: Row(children: [
        Container(width: 46, height: 46,
          decoration: BoxDecoration(
            gradient: const LinearGradient(colors: [_pinkLight, _pink],
                begin: Alignment.topLeft, end: Alignment.bottomRight),
            borderRadius: BorderRadius.circular(14),
            boxShadow: [BoxShadow(color: _pink.withOpacity(.3), blurRadius: 8, offset: const Offset(0, 3))]),
          child: const Icon(Icons.water_drop_rounded, color: Colors.white, size: 22)),
        const SizedBox(width: 14),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text('${_mn(e.startDate.month)} ${e.startDate.year}',
                style: const TextStyle(color: _dark, fontSize: 14,
                    fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
            const SizedBox(width: 6),
            if (isLatest) _chip('Current', _pink),
            if (e.isHistorical && !isLatest) _chip('History', _purple),
            if (e.unusualBleeding) ...[const SizedBox(width: 4), _chip('⚠ Unusual', _amber)],
          ]),
          const SizedBox(height: 3),
          Text('${e.startDate.day} ${_sn(e.startDate.month)}  •  ${e.cycleLength}d cycle  •  ${e.periodDuration}d period',
              style: TextStyle(color: Colors.grey[500], fontSize: 12, decoration: TextDecoration.none)),
        ])),
        Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: _pink, borderRadius: BorderRadius.circular(8)),
            child: Text('${e.periodDuration}d', style: const TextStyle(color: Colors.white,
                fontSize: 11, fontWeight: FontWeight.w700, decoration: TextDecoration.none))),
          const SizedBox(height: 4),
          Text('${e.cycleLength}d cycle', style: const TextStyle(color: _purple,
              fontSize: 11, fontWeight: FontWeight.w500, decoration: TextDecoration.none)),
        ]),
        const SizedBox(width: 8),
        // Edit button
        GestureDetector(
          onTap: () => _showLogSheet(edit: e),
          child: Container(padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(color: const Color(0xFFFFF0F7), borderRadius: BorderRadius.circular(9)),
            child: const Icon(Icons.edit_rounded, color: _pink, size: 15))),
        const SizedBox(width: 6),
        // Delete button
        GestureDetector(
          onTap: () async {
            final ok = await _confirm('Delete Cycle',
                'Remove cycle starting ${_fmt(e.startDate)}?');
            if (!ok) return;
            setState(() => _history.removeWhere((h) => _sameDay(h.startDate, e.startDate)));
            await _Cache.saveHistory(_history);
            if (e.id != null) await _Api.deleteCycle(e.id!, _userId);
            _snack('🗑 Cycle removed');
          },
          child: Container(padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              color: _pink.withOpacity(.08), borderRadius: BorderRadius.circular(9)),
            child: const Icon(Icons.delete_outline_rounded, color: _pink, size: 15))),
      ]));
  }

  Widget _chip(String l, Color c) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
    decoration: BoxDecoration(color: c, borderRadius: BorderRadius.circular(8)),
    child: Text(l, style: const TextStyle(color: Colors.white, fontSize: 9,
        fontWeight: FontWeight.w700, decoration: TextDecoration.none)));

  // ── insights tab ──────────────────────────────────────────────────────────
  Widget _insightsTab() {
    if (_history.isEmpty) {
      return Column(mainAxisSize: MainAxisSize.min, children: const [
        Text('📊', style: TextStyle(fontSize: 48, decoration: TextDecoration.none)),
        SizedBox(height: 12),
        Text('Log a few cycles to see insights', textAlign: TextAlign.center,
            style: TextStyle(color: _mid, fontSize: 13, decoration: TextDecoration.none)),
      ]);
    }
    final lens  = _history.map((e) => e.cycleLength).toList();
    final avg   = lens.reduce((a, b) => a + b) / lens.length;
    final diff  = lens.reduce(math.max) - lens.reduce(math.min);
    final isReg = diff <= 3;
    final avgPd = _history.map((e) => e.periodDuration).reduce((a, b) => a + b) / _history.length;
    final unc   = _history.where((e) => e.unusualBleeding).length;

    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      if (_mlPred != null) ...[
        Container(padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            gradient: const LinearGradient(colors: [_pinkLight, _pink],
                begin: Alignment.topLeft, end: Alignment.bottomRight),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [BoxShadow(color: _pink.withOpacity(.2), blurRadius: 12, offset: const Offset(0, 4))]),
          child: Row(children: [
            const Icon(Icons.auto_awesome_rounded, color: Colors.white, size: 22),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('Smart Predictions On', style: TextStyle(color: Colors.white, fontSize: 13,
                  fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
              Text('Based on ${_history.length} logged cycle${_history.length == 1 ? "" : "s"}  •  Ovulation: day ${_mlPred!.ovulationDay}',
                  style: const TextStyle(color: Colors.white70, fontSize: 11, decoration: TextDecoration.none)),
            ])),
          ])),
        const SizedBox(height: 14),
      ],
      const Text('Cycle Statistics', style: TextStyle(color: _dark, fontSize: 14,
          fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
      const SizedBox(height: 10),
      Row(children: [
        Expanded(child: _iStat(avg.toStringAsFixed(1), 'Avg Cycle', _pink)),
        const SizedBox(width: 10),
        Expanded(child: _iStat('${avgPd.toStringAsFixed(1)}d', 'Avg Period', _purple)),
        const SizedBox(width: 10),
        Expanded(child: _iStat('${_history.length}', 'Logged', _teal)),
      ]),
      const SizedBox(height: 16),
      const Text('Cycle Length Trend', style: TextStyle(color: _dark, fontSize: 14,
          fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
      const SizedBox(height: 10),
      _trendChart(),
      const SizedBox(height: 14),
      _iCard(isReg ? Icons.check_circle_rounded : Icons.warning_rounded,
          isReg ? _teal : _amber,
          isReg ? 'Regular Cycles' : 'Cycle Variability',
          isReg ? 'Variation ≤3 days — healthy pattern.'
              : 'Cycle varies by $diff days. Consider discussing with your doctor.'),
      const SizedBox(height: 10),
      _iCard(Icons.analytics_rounded, _purple, 'Prediction Accuracy',
          _history.length >= 3
              ? 'ML model active with ${_history.length} cycles logged.'
              : 'Log ${3 - _history.length} more cycle(s) to activate ML.'),
      const SizedBox(height: 10),
      if (unc > 0) ...[
        _iCard(Icons.warning_amber_rounded, _amber, 'Unusual Bleeding',
            '$unc of ${_history.length} cycles flagged. Consider consulting a doctor.'),
        const SizedBox(height: 10),
      ],
      _iCard(Icons.monitor_heart_rounded, _pink, 'Health Connection',
          'Cycle data improves PCOD, endometriosis & cervical cancer prediction accuracy.'),

      // Symptom patterns — only show if user has logged symptoms
      if (_dailyLogs.isNotEmpty) ...[
        const SizedBox(height: 14),
        const Text('Symptom Patterns', style: TextStyle(color: _dark, fontSize: 14,
            fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
        const SizedBox(height: 10),
        _symptomPatterns(),
      ],
    ]);
  }

  Widget _symptomPatterns() {
    // Count symptom frequency across all logged days
    final moodCount   = <String, int>{};
    final physCount   = <String, int>{};
    int heavyDays = 0, crampDays = 0;

    for (final log in _dailyLogs.values) {
      if (log.mood.isNotEmpty) moodCount[log.mood] = (moodCount[log.mood] ?? 0) + 1;
      for (final s in log.symptoms) physCount[s] = (physCount[s] ?? 0) + 1;
      if (log.flow == 'heavy') heavyDays++;
      if (log.symptoms.contains('cramps')) crampDays++;
    }

    final topMood = moodCount.entries.isEmpty ? null
        : moodCount.entries.reduce((a, b) => a.value > b.value ? a : b);
    final topPhys = physCount.entries.isEmpty ? null
        : physCount.entries.reduce((a, b) => a.value > b.value ? a : b);
    final total = _dailyLogs.length;

    return Column(mainAxisSize: MainAxisSize.min, children: [
      // Mood stat
      if (topMood != null)
        _iCard(Icons.mood_rounded, _purple, 'Most Common Mood',
            '${_moodEmoji(topMood.key)} ${_cap(topMood.key)} logged on ${topMood.value} of $total days tracked.'),
      if (topMood != null) const SizedBox(height: 10),

      // Physical symptom stat
      if (topPhys != null)
        _iCard(Icons.healing_rounded, _amber, 'Most Common Symptom',
            '${_physEmoji(topPhys.key)} ${_cap(topPhys.key)} on ${topPhys.value} of $total days tracked.'),
      if (topPhys != null) const SizedBox(height: 10),

      // Flow pattern
      if (heavyDays > 0)
        _iCard(Icons.water_drop_rounded, _pink, 'Flow Pattern',
            'Heavy flow recorded on $heavyDays day${heavyDays > 1 ? "s" : ""}. '
            '${heavyDays > 3 ? "Consistently heavy periods may be worth discussing with a doctor." : "This appears within normal range."}'),
      if (heavyDays > 0) const SizedBox(height: 10),

      if (total < 3)
        _iCard(Icons.insights_rounded, _mid, 'Keep Logging',
            'Log symptoms on more days to see your personal patterns here.'),
    ]);
  }

  String _moodEmoji(String m) =>
      {'happy':'😊','sad':'😢','anxious':'😰','irritable':'😤','calm':'😌'}[m] ?? '😐';
  String _physEmoji(String s) =>
      {'cramps':'😣','headache':'🤕','bloating':'😮','fatigue':'😴'}[s] ?? '•';
  String _cap(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);

  Widget _iStat(String v, String l, Color c) => Container(
    padding: const EdgeInsets.symmetric(vertical: 14),
    decoration: BoxDecoration(color: c.withOpacity(.08), borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.withOpacity(.2))),
    child: Column(children: [
      Text(v, style: TextStyle(color: c, fontSize: 20, fontWeight: FontWeight.w900,
          decoration: TextDecoration.none)),
      const SizedBox(height: 4),
      Text(l, style: const TextStyle(color: _mid, fontSize: 11, decoration: TextDecoration.none)),
    ]));

  Widget _trendChart() {
    final data = _history.reversed.take(6).toList();
    if (data.length < 2) {
      return const Padding(padding: EdgeInsets.all(8),
          child: Text('Log 2+ cycles to see trend.',
              style: TextStyle(color: _mid, fontSize: 12, decoration: TextDecoration.none)));
    }
    final maxL = data.map((e) => e.cycleLength).reduce(math.max).toDouble();
    final avgL = data.map((e) => e.cycleLength).reduce((a,b) => a+b) / data.length;
    final avgH = (avgL / maxL) * 70;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
      child: Column(children: [
        SizedBox(height: 100, child: Stack(children: [
          // Average line
          Positioned(
            bottom: 14 + avgH,
            left: 0, right: 0,
            child: Row(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                decoration: BoxDecoration(color: _teal.withOpacity(.15),
                    borderRadius: BorderRadius.circular(4)),
                child: Text('avg ${avgL.toStringAsFixed(0)}d',
                    style: const TextStyle(color: _teal, fontSize: 9,
                        fontWeight: FontWeight.w700, decoration: TextDecoration.none))),
              Expanded(child: Container(height: 1,
                  color: _teal.withOpacity(.4),
                  margin: const EdgeInsets.only(left: 4))),
            ])),
          // Bars
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: data.map((e) {
              final h = (e.cycleLength / maxL) * 70;
              final isLast = data.indexOf(e) == data.length - 1;
              final isAboveAvg = e.cycleLength > avgL;
              return Column(mainAxisAlignment: MainAxisAlignment.end, children: [
                Text('${e.cycleLength}', style: TextStyle(
                    color: isLast ? _pink : _mid, fontSize: 10,
                    fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
                const SizedBox(height: 3),
                AnimatedContainer(
                  duration: const Duration(milliseconds: 600),
                  width: 26, height: h,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.bottomCenter, end: Alignment.topCenter,
                      colors: isLast
                          ? [_pink, _pinkLight]
                          : isAboveAvg
                              ? [_purple.withOpacity(.6), _purple.withOpacity(.3)]
                              : [_purple.withOpacity(.35), _purple.withOpacity(.15)]),
                    borderRadius: const BorderRadius.vertical(top: Radius.circular(6)))),
                const SizedBox(height: 4),
                Text(_sn(e.startDate.month), style: const TextStyle(
                    color: _mid, fontSize: 9, decoration: TextDecoration.none)),
              ]);
            }).toList()),
        ])),
        const SizedBox(height: 8),
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          _ldot(_pink, 'Latest'),
          const SizedBox(width: 12),
          _ldot(_purple.withOpacity(.5), 'Past'),
          const SizedBox(width: 12),
          _ldot(_teal, 'Average'),
        ]),
      ]));
  }

  Widget _iCard(IconData icon, Color c, String title, String body) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.withOpacity(.15), width: 1.5),
        boxShadow: [BoxShadow(color: c.withOpacity(.05), blurRadius: 8, offset: const Offset(0, 3))]),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Container(padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(color: c.withOpacity(.12), borderRadius: BorderRadius.circular(10)),
          child: Icon(icon, color: c, size: 16)),
      const SizedBox(width: 10),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title, style: const TextStyle(color: _dark, fontSize: 12,
            fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
        const SizedBox(height: 3),
        Text(body, style: const TextStyle(color: _mid, fontSize: 11,
            height: 1.4, decoration: TextDecoration.none)),
      ])),
    ]));

  // ── sheet / modal openers ─────────────────────────────────────────────────
  Future<void> _showSymptomSheet(String date, DailyLog? existing) async {
    if (!mounted) return;
    await showModalBottomSheet(
      context: context, isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _SymptomSheet(
        date: date, existing: existing,
        onSave: (log) async { await _saveSymptoms(log); }));
  }

  Future<void> _showLogSheet({bool isHistory = false, CycleEntry? edit}) async {
    if (!mounted) return;
    await showModalBottomSheet(context: context, isScrollControlled: true,
        backgroundColor: Colors.transparent,
        builder: (_) => _LogSheet(
          isHistory: isHistory,
          edit: edit,
          onSave: _saveCycle,
          onDelete: edit != null ? _deleteCycle : null));
  }

  void _showCalModal() => showModalBottomSheet(
    context: context, isScrollControlled: true,
    backgroundColor: Colors.transparent, enableDrag: true,
    builder: (_) => DraggableScrollableSheet(
      initialChildSize: .78, minChildSize: .45, maxChildSize: .95, expand: false,
      builder: (ctx, scroll) => _FullCalModal(
        selDate: _selDate,
        history: _history,           // pass full history — modal computes own colours
        scrollController: scroll,
        onDayTap: (d) => setState(() => _selDay = d),
        onMonthChanged: (d) => setState(() { _selDate = d; _selDay = null; }))));

  void _showNotifPanel() {
    setState(() => _hasUnread = false);
    showModalBottomSheet(context: context, isScrollControlled: true,
        backgroundColor: Colors.transparent,
        builder: (_) => _NotifPanel(
          cd: _latest, p: _nPeriod, f: _nFertile, m: _nMed, i: _nInsights,
          onChanged: (p, f, m, i) => setState(() {
            _nPeriod = p; _nFertile = f; _nMed = m; _nInsights = i;
          })));
  }

  // ── utils ─────────────────────────────────────────────────────────────────
  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: const TextStyle(fontWeight: FontWeight.w600, color: Colors.white)),
      backgroundColor: _pink, behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      margin: const EdgeInsets.all(16), duration: const Duration(seconds: 3)));
  }

  Future<bool> _confirm(String title, String msg) async =>
      await showDialog<bool>(context: context, builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text(title, style: const TextStyle(color: _dark, fontWeight: FontWeight.w800)),
        content: Text(msg, style: const TextStyle(color: _mid)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel', style: TextStyle(color: _mid))),
          TextButton(onPressed: () => Navigator.pop(context, true),
              child: const Text('Delete', style: TextStyle(color: _pink, fontWeight: FontWeight.w700))),
        ])) ?? false;

  bool   _sameDay(DateTime a, DateTime b) => a.year==b.year && a.month==b.month && a.day==b.day;
  String _fmt(DateTime d) => '${d.day} ${_mn(d.month)} ${d.year}';
  String _dayName(DateTime d) => ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday-1];
  String _mn(int m) => const ['January','February','March','April','May','June',
      'July','August','September','October','November','December'][m-1];
  String _sn(int m) => const ['Jan','Feb','Mar','Apr','May','Jun',
      'Jul','Aug','Sep','Oct','Nov','Dec'][m-1];
}

// ─────────────────────────── LOG SHEET ───────────────────────────────────────

class _LogSheet extends StatefulWidget {
  final bool isHistory;
  final CycleEntry? edit;
  final void Function(CycleEntry) onSave;
  final void Function(CycleEntry)? onDelete;
  const _LogSheet({required this.isHistory, required this.onSave, this.edit, this.onDelete});
  @override
  State<_LogSheet> createState() => _LogSheetState();
}

class _LogSheetState extends State<_LogSheet> {
  late DateTime _date;
  late int      _cl;
  late int      _pd;
  late bool     _hist;
  late bool     _unusual;
  late TextEditingController _pdCtrl;

  @override
  void initState() {
    super.initState();
    final e  = widget.edit;
    _date    = e?.startDate       ?? DateTime.now();
    _cl      = e?.cycleLength     ?? 28;
    _pd      = e?.periodDuration  ?? 5;
    _hist    = e?.isHistorical    ?? widget.isHistory;
    _unusual = e?.unusualBleeding ?? false;
    _pdCtrl  = TextEditingController(text: '$_pd');
  }

  @override
  void dispose() { _pdCtrl.dispose(); super.dispose(); }

  String _mn(int m) => const ['January','February','March','April','May','June',
      'July','August','September','October','November','December'][m-1];
  String _sn(int m) => const ['Jan','Feb','Mar','Apr','May','Jun',
      'Jul','Aug','Sep','Oct','Nov','Dec'][m-1];

  @override
  Widget build(BuildContext context) {
    final prev = _date.add(Duration(days: _cl));
    final ovul = _date.add(Duration(days: _cl - 14));

    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFFFFF5F8),
        borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

          // handle
          Center(child: Container(
            margin: const EdgeInsets.only(top: 12, bottom: 20),
            width: 44, height: 5,
            decoration: BoxDecoration(
              color: const Color(0xFFE0C8D8), borderRadius: BorderRadius.circular(3)))),

          // title
          Row(children: [
            Container(padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(12),
                boxShadow: [BoxShadow(color: _pink.withOpacity(.3), blurRadius: 8, offset: const Offset(0, 3))]),
              child: Icon(widget.edit != null ? Icons.edit_rounded : Icons.water_drop_rounded,
                  color: Colors.white, size: 20)),
            const SizedBox(width: 12),
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(widget.edit != null ? 'Edit Cycle' : _hist ? 'Add Past Cycle' : 'Log Current Cycle',
                  style: const TextStyle(color: _dark, fontSize: 18,
                      fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
              Text(_hist ? 'Enter a past cycle start date' : 'When did this period start?',
                  style: const TextStyle(color: _mid, fontSize: 12, decoration: TextDecoration.none)),
            ]),
          ]),
          const SizedBox(height: 20),

          // current / past toggle
          if (widget.edit == null) ...[
            Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFFFCE7F3))),
              child: Row(children: [
                Expanded(child: GestureDetector(
                  onTap: () => setState(() => _hist = false),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    decoration: BoxDecoration(
                      gradient: !_hist ? const LinearGradient(colors: [_pinkLight, _pink]) : null,
                      borderRadius: BorderRadius.circular(10)),
                    child: Center(child: Text('Current Cycle',
                        style: TextStyle(color: !_hist ? Colors.white : _mid,
                            fontSize: 13, fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none)))))),
                Expanded(child: GestureDetector(
                  onTap: () => setState(() => _hist = true),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    decoration: BoxDecoration(
                      gradient: _hist ? const LinearGradient(colors: [_pinkLight, _pink]) : null,
                      borderRadius: BorderRadius.circular(10)),
                    child: Center(child: Text('Past Cycle',
                        style: TextStyle(color: _hist ? Colors.white : _mid,
                            fontSize: 13, fontWeight: FontWeight.w700,
                            decoration: TextDecoration.none)))))),
              ])),
            const SizedBox(height: 20),
          ],

          // field 1: period start date
          const _Lbl('Period Start Date'),
          const SizedBox(height: 8),
          GestureDetector(
            onTap: () async {
              final last = _hist
                  ? DateTime.now().subtract(const Duration(days: 1))
                  : DateTime.now();
              final p = await showDatePicker(
                context: context,
                initialDate: _date.isAfter(last) ? last : _date,
                firstDate: DateTime(2020), lastDate: last,
                builder: (ctx, child) => Theme(
                  data: Theme.of(ctx).copyWith(colorScheme: const ColorScheme.light(
                      primary: _pink, onPrimary: Colors.white, surface: Colors.white)),
                  child: child!));
              if (p != null) setState(() => _date = p);
            },
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: _pink.withOpacity(.3), width: 1.5)),
              child: Row(children: [
                const Icon(Icons.calendar_today_rounded, color: _pink, size: 20),
                const SizedBox(width: 12),
                Text('${_date.day} ${_mn(_date.month)} ${_date.year}',
                    style: const TextStyle(color: _dark, fontSize: 15,
                        fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
                const Spacer(),
                const Icon(Icons.chevron_right_rounded, color: Color(0xFFD4A0B8)),
              ]))),
          const SizedBox(height: 20),

          // field 2: cycle length
          const _Lbl('Cycle Length (days)'),
          const SizedBox(height: 4),
          Text('Average cycle is 28 days. Typical range: 21–35 days.',
              style: TextStyle(color: Colors.grey[500], fontSize: 11, decoration: TextDecoration.none)),
          const SizedBox(height: 8),
          _Stepper(value: _cl, min: 15, max: 90, onChanged: (v) => setState(() => _cl = v)),
          // Warning for unusual cycle length
          if (_cl < 21 || _cl > 45) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: _amber.withOpacity(.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _amber.withOpacity(.3))),
              child: Row(children: [
                const Icon(Icons.info_outline_rounded, color: _amber, size: 14),
                const SizedBox(width: 8),
                Expanded(child: Text(
                  _cl < 21
                      ? 'Cycles shorter than 21 days may need medical attention.'
                      : 'Cycles longer than 45 days can be associated with hormonal imbalances. Consider consulting a doctor.',
                  style: TextStyle(color: _amber.withOpacity(.9), fontSize: 11,
                      height: 1.4, decoration: TextDecoration.none))),
              ])),
          ],
          const SizedBox(height: 20),

          // field 3: period duration
          const _Lbl('Period Duration (days)'),
          const SizedBox(height: 4),
          Text('Most periods last 3–7 days. Enter your actual duration.',
              style: TextStyle(color: Colors.grey[500], fontSize: 11, decoration: TextDecoration.none)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
                border: Border.all(color: _pink.withOpacity(.3), width: 1.5)),
            child: TextField(
              controller: _pdCtrl,
              keyboardType: TextInputType.number,
              inputFormatters: [FilteringTextInputFormatter.digitsOnly],
              style: const TextStyle(color: _dark, fontSize: 15,
                  fontWeight: FontWeight.w700, decoration: TextDecoration.none),
              decoration: InputDecoration(
                border: InputBorder.none,
                hintText: 'e.g. 5',
                hintStyle: TextStyle(color: Colors.grey[400]),
                prefixIcon: const Icon(Icons.bloodtype_outlined, color: _pink, size: 20),
                suffixText: 'days',
                suffixStyle: const TextStyle(color: _mid, fontSize: 13)),
              onChanged: (v) {
                final parsed = int.tryParse(v);
                if (parsed != null && parsed >= 1 && parsed <= 20) setState(() => _pd = parsed);
              })),
          // Warning for unusual period duration
          if (_pd > 7) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: _amber.withOpacity(.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _amber.withOpacity(.3))),
              child: Row(children: [
                const Icon(Icons.info_outline_rounded, color: _amber, size: 14),
                const SizedBox(width: 8),
                Expanded(child: Text(
                  _pd > 10
                      ? 'Periods lasting more than 10 days (menorrhagia) should be discussed with a doctor.'
                      : 'Periods longer than 7 days are worth mentioning to your doctor.',
                  style: TextStyle(color: _amber.withOpacity(.9), fontSize: 11,
                      height: 1.4, decoration: TextDecoration.none))),
              ])),
          ],
          const SizedBox(height: 20),

          // field 4: unusual bleeding
          const _Lbl('Unusual Bleeding?'),
          const SizedBox(height: 4),
          Text('Spotting, heavier than normal, or unexpected bleeding',
              style: TextStyle(color: Colors.grey[500], fontSize: 11, decoration: TextDecoration.none)),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            decoration: BoxDecoration(
              color: _unusual ? _amber.withOpacity(.06) : Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: _unusual ? _amber.withOpacity(.4) : const Color(0xFFFCE7F3), width: 1.5)),
            child: Row(children: [
              Container(padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: (_unusual ? _amber : _pink).withOpacity(.12),
                  borderRadius: BorderRadius.circular(9)),
                child: Icon(_unusual ? Icons.warning_amber_rounded : Icons.bloodtype_outlined,
                    color: _unusual ? _amber : _pink, size: 18)),
              const SizedBox(width: 12),
              Expanded(child: Text(_unusual ? 'Yes — unusual bleeding noted' : 'No — normal bleeding',
                  style: TextStyle(color: _unusual ? _amber : _dark, fontSize: 13,
                      fontWeight: FontWeight.w600, decoration: TextDecoration.none))),
              Switch(
                value: _unusual,
                onChanged: (v) => setState(() => _unusual = v),
                activeColor: _amber, activeTrackColor: _amber.withOpacity(.25),
                inactiveThumbColor: Colors.grey.shade300,
                inactiveTrackColor: Colors.grey.shade100,
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap),
            ])),
          if (_unusual) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: _amber.withOpacity(.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: _amber.withOpacity(.25))),
              child: Row(children: [
                const Icon(Icons.info_outline_rounded, color: _amber, size: 14),
                const SizedBox(width: 8),
                Expanded(child: Text('Flagged in ML prediction and Insights.',
                    style: TextStyle(color: _amber.withOpacity(.8), fontSize: 11,
                        height: 1.4, decoration: TextDecoration.none))),
              ])),
          ],
          const SizedBox(height: 24),

          // prediction preview
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('🔮 Prediction Preview', style: TextStyle(color: _dark,
                  fontSize: 13, fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
              const SizedBox(height: 4),
              Text('Dates update after saving',
                  style: TextStyle(color: Colors.grey[400], fontSize: 10, decoration: TextDecoration.none)),
              const SizedBox(height: 10),
              _pr('Next Period',    '${prev.day} ${_sn(prev.month)} ${prev.year}'),
              _pr('Ovulation',      '${ovul.day} ${_sn(ovul.month)}'),
              _pr('Fertile Window', '${ovul.subtract(const Duration(days: 5)).day}–${ovul.day} ${_sn(ovul.month)}'),
              _pr('Period Duration', '$_pd days'),
            ])),
          const SizedBox(height: 24),

          // save button
          GestureDetector(
            onTap: () {
              final parsed = int.tryParse(_pdCtrl.text);
              final finalPd = (parsed != null && parsed >= 1 && parsed <= 20) ? parsed : _pd;
              widget.onSave(CycleEntry(
                id: widget.edit?.id, startDate: _date, cycleLength: _cl,
                periodDuration: finalPd, isHistorical: _hist, unusualBleeding: _unusual));
              Navigator.pop(context);
            },
            child: Container(
              width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [BoxShadow(color: _pink.withOpacity(.35), blurRadius: 14, offset: const Offset(0, 5))]),
              child: Center(child: Text(
                widget.edit != null ? 'Update Cycle' : 'Save & Predict',
                style: const TextStyle(color: Colors.white, fontSize: 16,
                    fontWeight: FontWeight.w700, decoration: TextDecoration.none))))),

          // Delete button — only shown when editing an existing cycle
          if (widget.edit != null && widget.onDelete != null) ...[
            const SizedBox(height: 12),
            GestureDetector(
              onTap: () async {
                Navigator.pop(context);
                widget.onDelete!(widget.edit!);
              },
              child: Container(
                width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 14),
                decoration: BoxDecoration(
                  color: _pink.withOpacity(.06),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: _pink.withOpacity(.3), width: 1.5)),
                child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Icon(Icons.delete_outline_rounded, color: _pink.withOpacity(.8), size: 18),
                  const SizedBox(width: 8),
                  Text('Delete This Cycle', style: TextStyle(color: _pink.withOpacity(.8),
                      fontSize: 14, fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
                ]))),
          ],
        ])));
  }

  Widget _pr(String l, String v) => Padding(
    padding: const EdgeInsets.only(bottom: 5),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(l, style: const TextStyle(color: Color(0xFFAA99BB), fontSize: 12, decoration: TextDecoration.none)),
      Text(v, style: const TextStyle(color: _pink, fontSize: 12,
          fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
    ]));
}

// ─────────────────────────── SYMPTOM SHEET ──────────────────────────────────

class _SymptomSheet extends StatefulWidget {
  final String    date;
  final DailyLog? existing;
  final Future<void> Function(DailyLog) onSave;
  const _SymptomSheet({required this.date, required this.onSave, this.existing});
  @override
  State<_SymptomSheet> createState() => _SymptomSheetState();
}

class _SymptomSheetState extends State<_SymptomSheet> {
  String       _mood = '';
  String       _flow = 'none';
  Set<String>  _syms = {};
  bool         _saving = false;

  static const _moods = [
    {'key':'happy',    'label':'Happy',    'emoji':'😊'},
    {'key':'calm',     'label':'Calm',     'emoji':'😌'},
    {'key':'sad',      'label':'Sad',      'emoji':'😢'},
    {'key':'anxious',  'label':'Anxious',  'emoji':'😰'},
    {'key':'irritable','label':'Irritable','emoji':'😤'},
  ];

  static const _flows = [
    {'key':'none',  'label':'None',   'emoji':'○'},
    {'key':'light', 'label':'Light',  'emoji':'💧'},
    {'key':'medium','label':'Medium', 'emoji':'💧💧'},
    {'key':'heavy', 'label':'Heavy',  'emoji':'💧💧💧'},
  ];

  static const _physical = [
    {'key':'cramps',   'label':'Cramps',   'emoji':'😣'},
    {'key':'headache', 'label':'Headache', 'emoji':'🤕'},
    {'key':'bloating', 'label':'Bloating', 'emoji':'😮‍💨'},
    {'key':'fatigue',  'label':'Fatigue',  'emoji':'😴'},
  ];

  @override
  void initState() {
    super.initState();
    if (widget.existing != null) {
      _mood = widget.existing!.mood;
      _flow = widget.existing!.flow;
      _syms = Set.from(widget.existing!.symptoms);
    }
  }

  String _fmt(String d) {
    // yyyy-MM-dd → "8 May 2026"
    final parts = d.split('-');
    if (parts.length != 3) return d;
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    final m = int.tryParse(parts[1]);
    return '${int.parse(parts[2])} ${m != null ? months[m-1] : parts[1]} ${parts[0]}';
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFFFFF5F8),
        borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(child: Column(
        crossAxisAlignment: CrossAxisAlignment.start, children: [

        // Handle
        Center(child: Container(margin: const EdgeInsets.only(top: 12, bottom: 20),
            width: 44, height: 5, decoration: BoxDecoration(
                color: const Color(0xFFE0C8D8), borderRadius: BorderRadius.circular(3)))),

        // Title
        Row(children: [
          Container(padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_pinkLight, _pink]),
              borderRadius: BorderRadius.circular(12),
              boxShadow: [BoxShadow(color: _pink.withOpacity(.3), blurRadius: 8, offset: const Offset(0, 3))]),
            child: const Icon(Icons.favorite_rounded, color: Colors.white, size: 20)),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('How are you feeling?', style: TextStyle(color: _dark, fontSize: 18,
                fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
            Text(_fmt(widget.date), style: const TextStyle(color: _mid, fontSize: 12,
                decoration: TextDecoration.none)),
          ]),
        ]),
        const SizedBox(height: 24),

        // ── Mood ────────────────────────────────────────────────────────────
        const _Lbl('Mood'),
        const SizedBox(height: 10),
        Wrap(spacing: 8, runSpacing: 8, children: _moods.map((m) {
          final sel = _mood == m['key'];
          return GestureDetector(
            onTap: () => setState(() => _mood = sel ? '' : m['key']!),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
              decoration: BoxDecoration(
                color: sel ? _purple : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: sel ? _purple : const Color(0xFFE5D4F0), width: 1.5),
                boxShadow: sel ? [BoxShadow(color: _purple.withOpacity(.25),
                    blurRadius: 8, offset: const Offset(0, 3))] : null),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Text(m['emoji']!, style: const TextStyle(fontSize: 16, decoration: TextDecoration.none)),
                const SizedBox(width: 6),
                Text(m['label']!, style: TextStyle(color: sel ? Colors.white : _dark,
                    fontSize: 13, fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
              ])));
        }).toList()),
        const SizedBox(height: 22),

        // ── Flow ────────────────────────────────────────────────────────────
        const _Lbl('Flow Intensity'),
        const SizedBox(height: 10),
        Row(children: _flows.map((f) {
          final sel = _flow == f['key'];
          return Expanded(child: GestureDetector(
            onTap: () => setState(() => _flow = f['key']!),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              margin: const EdgeInsets.symmetric(horizontal: 4),
              padding: const EdgeInsets.symmetric(vertical: 12),
              decoration: BoxDecoration(
                color: sel ? _pink : Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: sel ? _pink : const Color(0xFFE5D4F0), width: 1.5),
                boxShadow: sel ? [BoxShadow(color: _pink.withOpacity(.25),
                    blurRadius: 8, offset: const Offset(0, 3))] : null),
              child: Column(children: [
                Text(f['emoji']!, style: const TextStyle(fontSize: 14, decoration: TextDecoration.none)),
                const SizedBox(height: 4),
                Text(f['label']!, style: TextStyle(color: sel ? Colors.white : _mid,
                    fontSize: 11, fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
              ]))));
        }).toList()),
        const SizedBox(height: 22),

        // ── Physical symptoms ────────────────────────────────────────────────
        const _Lbl('Physical Symptoms'),
        const SizedBox(height: 4),
        Text('Select all that apply', style: TextStyle(color: Colors.grey[500], fontSize: 11,
            decoration: TextDecoration.none)),
        const SizedBox(height: 10),
        Wrap(spacing: 8, runSpacing: 8, children: _physical.map((p) {
          final sel = _syms.contains(p['key']);
          return GestureDetector(
            onTap: () => setState(() {
              if (sel) _syms.remove(p['key']); else _syms.add(p['key']!);
            }),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
              decoration: BoxDecoration(
                color: sel ? _amber : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: sel ? _amber : const Color(0xFFE5D4F0), width: 1.5),
                boxShadow: sel ? [BoxShadow(color: _amber.withOpacity(.25),
                    blurRadius: 8, offset: const Offset(0, 3))] : null),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Text(p['emoji']!, style: const TextStyle(fontSize: 16, decoration: TextDecoration.none)),
                const SizedBox(width: 6),
                Text(p['label']!, style: TextStyle(color: sel ? Colors.white : _dark,
                    fontSize: 13, fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
              ])));
        }).toList()),
        const SizedBox(height: 28),

        // ── Save button ──────────────────────────────────────────────────────
        GestureDetector(
          onTap: _saving ? null : () async {
            setState(() => _saving = true);
            await widget.onSave(DailyLog(
              date: widget.date, mood: _mood,
              flow: _flow, symptoms: _syms.toList()));
            if (mounted) Navigator.pop(context);
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 16),
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_pinkLight, _pink]),
              borderRadius: BorderRadius.circular(18),
              boxShadow: [BoxShadow(color: _pink.withOpacity(.35),
                  blurRadius: 14, offset: const Offset(0, 5))]),
            child: Center(child: _saving
                ? const SizedBox(width: 20, height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Text('Save', style: TextStyle(color: Colors.white, fontSize: 16,
                    fontWeight: FontWeight.w700, decoration: TextDecoration.none))))),
      ])));
  }
}

// ─────────────────────────── FULL CAL MODAL ──────────────────────────────────

class _FullCalModal extends StatefulWidget {
  final DateTime selDate;
  final List<CycleEntry> history;  // full history so modal computes own colours
  final Function(int)      onDayTap;
  final Function(DateTime) onMonthChanged;
  final ScrollController   scrollController;
  const _FullCalModal({
    required this.selDate, required this.history,
    required this.onDayTap, required this.onMonthChanged,
    required this.scrollController});
  @override
  State<_FullCalModal> createState() => _FullCalModalState();
}

class _FullCalModalState extends State<_FullCalModal> {
  late DateTime _cur;
  int? _sel;

  @override void initState() { super.initState(); _cur = widget.selDate; }

  int get _days => DateTime(_cur.year, _cur.month + 1, 0).day;
  int get _off  => DateTime(_cur.year, _cur.month, 1).weekday - 1;

  // Compute colours fresh for _cur month from full history
  List<int> get _perDays {
    final s = <int>{};
    for (final e in widget.history) s.addAll(e.periodDaysForMonth(_cur.year, _cur.month));
    return s.toList();
  }
  List<int> get _ferDays {
    final s = <int>{};
    for (final e in widget.history) s.addAll(e.fertileDaysForMonth(_cur.year, _cur.month));
    return s.toList();
  }
  List<int> get _pmsDays {
    final s = <int>{};
    for (final e in widget.history) s.addAll(e.pmsDaysForMonth(_cur.year, _cur.month));
    return s.toList();
  }
  int? get _ovulDay {
    for (final e in widget.history) {
      final d = e.ovulDayForMonth(_cur.year, _cur.month);
      if (d != null) return d;
    }
    return null;
  }

  String _mn(int m) => const ['January','February','March','April','May','June',
      'July','August','September','October','November','December'][m-1];

  @override
  Widget build(BuildContext context) {
    final perDays = _perDays;
    final ferDays = _ferDays;
    final pmsDays = _pmsDays;
    final ovulDay = _ovulDay;

    return Container(
      decoration: const BoxDecoration(color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      child: ListView(controller: widget.scrollController, padding: EdgeInsets.zero,
        physics: const ClampingScrollPhysics(), children: [
        Center(child: Container(margin: const EdgeInsets.only(top: 12, bottom: 4),
            width: 44, height: 5, decoration: BoxDecoration(
                color: const Color(0xFFE0C8D8), borderRadius: BorderRadius.circular(3)))),
        Padding(padding: const EdgeInsets.fromLTRB(20, 12, 20, 4),
          child: Row(children: [
            Container(padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(12)),
              child: const Icon(Icons.calendar_month_rounded, color: Colors.white, size: 20)),
            const SizedBox(width: 12),
            const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Period Calendar', style: TextStyle(color: _dark, fontSize: 18,
                  fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
              Text('Tap a day to select it', style: TextStyle(color: _mid, fontSize: 12,
                  decoration: TextDecoration.none)),
            ])),
            GestureDetector(onTap: () => Navigator.pop(context),
              child: Container(padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: const Color(0xFFF5EEF5),
                    borderRadius: BorderRadius.circular(10)),
                child: const Icon(Icons.close_rounded, color: Color(0xFFBB8FAE), size: 18))),
          ])),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            _nb(Icons.chevron_left_rounded, () => setState(() {
              _cur = DateTime(_cur.year, _cur.month - 1, 1); _sel = null;
            })),
            Column(children: [
              Text(_mn(_cur.month), style: const TextStyle(color: _dark, fontSize: 20,
                  fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
              Text('${_cur.year}', style: const TextStyle(color: _mid, fontSize: 12,
                  fontWeight: FontWeight.w600, decoration: TextDecoration.none)),
            ]),
            _nb(Icons.chevron_right_rounded, () => setState(() {
              _cur = DateTime(_cur.year, _cur.month + 1, 1); _sel = null;
            })),
          ])),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(children: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].map((d) =>
            Expanded(child: Center(child: Text(d, style: TextStyle(
              color: (d=='Sat'||d=='Sun') ? const Color(0xFFE087A8) : _mid,
              fontSize: 11, fontWeight: FontWeight.w700,
              decoration: TextDecoration.none))))).toList())),
        const SizedBox(height: 8),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 12),
            child: _grid(perDays, ferDays, pmsDays, ovulDay)),
        const SizedBox(height: 12),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(color: const Color(0xFFFFF5F9),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFFEED4E0))),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
              _li(_pink, 'Period'), _li(_purple, 'Fertile'),
              _li(_teal, 'Ovulation'), _li(_amber, 'PMS'),
              _li(const Color(0xFFFFD6E8), 'Today'),
            ]))),
        Padding(
          padding: EdgeInsets.fromLTRB(20, 14, 20, MediaQuery.of(context).padding.bottom + 20),
          child: GestureDetector(
            onTap: () { widget.onMonthChanged(_cur); Navigator.pop(context); },
            child: Container(width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [BoxShadow(color: _pink.withOpacity(.35),
                    blurRadius: 14, offset: const Offset(0, 5))]),
              child: const Center(child: Text('Done', style: TextStyle(
                  color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700,
                  decoration: TextDecoration.none)))))),
      ]));
  }

  Widget _grid(List<int> perDays, List<int> ferDays, List<int> pmsDays, int? ovulDay) {
    final cells = [...List<int?>.filled(_off, null), ...List<int?>.generate(_days, (i) => i+1)];
    while (cells.length % 7 != 0) cells.add(null);
    return Column(children: List.generate(cells.length ~/ 7, (row) => Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(children: List.generate(7, (col) {
        final day = cells[row * 7 + col];
        if (day == null) return const Expanded(child: SizedBox());
        final isPer  = perDays.contains(day);
        final isFer  = ferDays.contains(day);
        final isOvul = ovulDay == day;
        final isPms  = pmsDays.contains(day);
        final isSel  = _sel == day;
        final isTod  = day == DateTime.now().day &&
            _cur.month == DateTime.now().month && _cur.year == DateTime.now().year;
        return Expanded(child: GestureDetector(
          onTap: () { setState(() => _sel = day); widget.onDayTap(day); },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            margin: const EdgeInsets.symmetric(horizontal: 2), height: 44,
            decoration: BoxDecoration(
              gradient: isPer
                  ? const LinearGradient(colors: [_pinkLight, _pink],
                        begin: Alignment.topLeft, end: Alignment.bottomRight)
                  : isOvul
                      ? const LinearGradient(colors: [Color(0xFF80D8CC), _teal],
                            begin: Alignment.topLeft, end: Alignment.bottomRight)
                      : isFer
                          ? const LinearGradient(colors: [Color(0xFFB5A4E0), _purple],
                                begin: Alignment.topLeft, end: Alignment.bottomRight)
                          : null,
              color: (!isPer && !isFer && !isOvul)
                  ? isPms  ? const Color(0xFFFFF3CD)
                  : isTod  ? const Color(0xFFFFD6E8)
                  : isSel  ? const Color(0xFFEED8F0)
                  : Colors.transparent : null,
              borderRadius: BorderRadius.circular(12),
              border: (isSel && !isPer) ? Border.all(color: _pink, width: 2)
                  : (isTod && !isPer)   ? Border.all(color: _pinkLight.withOpacity(.5), width: 1.5)
                  : null,
              boxShadow: isPer
                  ? [BoxShadow(color: _pink.withOpacity(.28), blurRadius: 6, offset: const Offset(0, 3))]
                  : (isFer || isOvul)
                      ? [BoxShadow(color: (isOvul ? _teal : _purple).withOpacity(.28),
                            blurRadius: 6, offset: const Offset(0, 3))]
                      : null),
            child: Center(child: Text('$day', style: TextStyle(
              color: isPer||isFer||isOvul ? Colors.white
                  : isTod ? _pink : isPms ? _amber
                  : col >= 5 ? const Color(0xFFE087A8) : _dark,
              fontSize: 14,
              fontWeight: (isPer||isFer||isOvul||isTod||isSel)
                  ? FontWeight.w700 : FontWeight.w500,
              decoration: TextDecoration.none))))));
      })))));
  }

  Widget _nb(IconData i, VoidCallback f) => GestureDetector(onTap: f,
    child: Container(width: 40, height: 40,
      decoration: BoxDecoration(color: const Color(0xFFFFF0F7),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFEEC4D6))),
      child: Icon(i, color: _pink, size: 22)));

  Widget _li(Color c, String l) => Row(children: [
    Container(width: 10, height: 10, decoration: BoxDecoration(color: c,
        borderRadius: BorderRadius.circular(3),
        boxShadow: [BoxShadow(color: c.withOpacity(.4), blurRadius: 4)])),
    const SizedBox(width: 4),
    Text(l, style: TextStyle(color: c, fontSize: 9, fontWeight: FontWeight.w600,
        decoration: TextDecoration.none))]);
}

// ─────────────────────────── NOTIF PANEL ─────────────────────────────────────

class _NotifPanel extends StatefulWidget {
  final CycleEntry? cd;
  final bool p, f, m, i;
  final void Function(bool, bool, bool, bool) onChanged;
  const _NotifPanel({required this.cd, required this.p, required this.f,
      required this.m, required this.i, required this.onChanged});
  @override
  State<_NotifPanel> createState() => _NotifPanelState();
}

class _NotifPanelState extends State<_NotifPanel> {
  late bool _p, _f, _m, _i;
  @override void initState() {
    super.initState(); _p=widget.p; _f=widget.f; _m=widget.m; _i=widget.i;
  }

  String _sm(int m) => const ['Jan','Feb','Mar','Apr','May','Jun',
      'Jul','Aug','Sep','Oct','Nov','Dec'][m-1];

  @override
  Widget build(BuildContext context) {
    final cd = widget.cd;
    return Container(
      decoration: const BoxDecoration(color: Color(0xFFFFF5F8),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      padding: EdgeInsets.fromLTRB(20, 0, 20, MediaQuery.of(context).padding.bottom + 24),
      child: SingleChildScrollView(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Center(child: Container(margin: const EdgeInsets.only(top: 12, bottom: 16),
            width: 44, height: 5, decoration: BoxDecoration(
                color: const Color(0xFFE0C8D8), borderRadius: BorderRadius.circular(3)))),
        Row(children: [
          Container(padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(12)),
            child: const Icon(Icons.notifications_rounded, color: Colors.white, size: 20)),
          const SizedBox(width: 12),
          const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Notifications', style: TextStyle(color: _dark, fontSize: 18,
                fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
            Text('Reminders & alerts', style: TextStyle(color: _mid, fontSize: 12,
                decoration: TextDecoration.none)),
          ])),
          GestureDetector(onTap: () => Navigator.pop(context),
            child: Container(padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(color: const Color(0xFFF5EEF5),
                  borderRadius: BorderRadius.circular(10)),
              child: const Icon(Icons.close_rounded, color: Color(0xFFBB8FAE), size: 18))),
        ]),
        if (cd != null) ...[
          const SizedBox(height: 20),
          const Text('Upcoming', style: TextStyle(color: _dark, fontSize: 14,
              fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
          const SizedBox(height: 10),
          _up(Icons.water_drop_rounded, _pink, 'Next Period',
              '${cd.nextPeriod.day} ${_sm(cd.nextPeriod.month)}',
              cd.daysUntilNext > 0 ? 'In ${cd.daysUntilNext}d' : 'Due!'),
          _up(Icons.favorite_rounded, _purple, 'Fertile Window',
              '${cd.fertileDays.first.day}–${cd.fertileDays.last.day}', '6 days'),
        ],
        const SizedBox(height: 20),
        const Text('Settings', style: TextStyle(color: _dark, fontSize: 14,
            fontWeight: FontWeight.w800, decoration: TextDecoration.none)),
        const SizedBox(height: 10),
        _tog('Period Reminder', 'Alert 2 days before expected period',
            Icons.water_drop_rounded, _pink, _p,
            (v) => setState(() { _p=v; widget.onChanged(_p,_f,_m,_i); })),
        _tog('Fertile Window', 'Notify when fertile days approach',
            Icons.favorite_rounded, _purple, _f,
            (v) => setState(() { _f=v; widget.onChanged(_p,_f,_m,_i); })),
        _tog('Medicine Reminder', 'Daily supplement reminder',
            Icons.medication_rounded, _teal, _m,
            (v) => setState(() { _m=v; widget.onChanged(_p,_f,_m,_i); })),
        _tog('Cycle Insights', 'Weekly cycle health summary',
            Icons.insights_rounded, _amber, _i,
            (v) => setState(() { _i=v; widget.onChanged(_p,_f,_m,_i); })),
        const SizedBox(height: 8),
        GestureDetector(onTap: () => Navigator.pop(context),
          child: Container(width: double.infinity, padding: const EdgeInsets.symmetric(vertical: 16),
            decoration: BoxDecoration(gradient: const LinearGradient(colors: [_pinkLight, _pink]),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [BoxShadow(color: _pink.withOpacity(.35), blurRadius: 14, offset: const Offset(0, 5))]),
            child: const Center(child: Text('Save & Close', style: TextStyle(color: Colors.white,
                fontSize: 16, fontWeight: FontWeight.w700, decoration: TextDecoration.none))))),
      ])));
  }

  Widget _up(IconData ic, Color c, String t, String s, String tr) =>
    Container(margin: const EdgeInsets.only(bottom: 8), padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.2)),
      child: Row(children: [
        Container(padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: c.withOpacity(.12), borderRadius: BorderRadius.circular(9)),
            child: Icon(ic, color: c, size: 16)),
        const SizedBox(width: 10),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(t, style: const TextStyle(color: _dark, fontSize: 12,
              fontWeight: FontWeight.w700, decoration: TextDecoration.none)),
          Text(s, style: const TextStyle(color: _mid, fontSize: 11, decoration: TextDecoration.none)),
        ])),
        Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: c.withOpacity(.10), borderRadius: BorderRadius.circular(20)),
            child: Text(tr, style: TextStyle(color: c, fontSize: 10,
                fontWeight: FontWeight.w700, decoration: TextDecoration.none))),
      ]));

  Widget _tog(String l, String s, IconData icon, Color c, bool v, ValueChanged<bool> fn) =>
    Container(margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: v ? c.withOpacity(.06) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: v ? c.withOpacity(.25) : const Color(0xFFFCE7F3), width: 1.2)),
      child: Row(children: [
        Container(padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: c.withOpacity(.12), borderRadius: BorderRadius.circular(9)),
            child: Icon(icon, color: c, size: 16)),
        const SizedBox(width: 10),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(l, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700,
              color: v ? _dark : Colors.grey.shade500, decoration: TextDecoration.none)),
          Text(s, style: const TextStyle(fontSize: 11, color: Color(0xFFCCBBDD),
              decoration: TextDecoration.none)),
        ])),
        Switch(value: v, onChanged: fn, activeColor: c, activeTrackColor: c.withOpacity(.25),
            inactiveThumbColor: Colors.grey.shade300, inactiveTrackColor: Colors.grey.shade100,
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap),
      ]));
}

// ─────────────────────────── SMALL HELPERS ───────────────────────────────────

class _Lbl extends StatelessWidget {
  final String text;
  const _Lbl(this.text);
  @override
  Widget build(BuildContext context) => Text(text, style: const TextStyle(
      color: _dark, fontSize: 13, fontWeight: FontWeight.w700,
      decoration: TextDecoration.none));
}

class _Stepper extends StatelessWidget {
  final int value, min, max;
  final ValueChanged<int> onChanged;
  const _Stepper({required this.value, required this.min,
      required this.max, required this.onChanged});
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5)),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      GestureDetector(
        onTap: value > min ? () => onChanged(value - 1) : null,
        child: Container(width: 36, height: 36,
          decoration: BoxDecoration(
            color: value > min ? const Color(0xFFFFF0F7) : Colors.grey.shade100,
            borderRadius: BorderRadius.circular(10)),
          child: Icon(Icons.remove_rounded,
              color: value > min ? _pink : Colors.grey.shade300, size: 18))),
      Text('$value', style: const TextStyle(color: _dark, fontSize: 20,
          fontWeight: FontWeight.w900, decoration: TextDecoration.none)),
      GestureDetector(
        onTap: value < max ? () => onChanged(value + 1) : null,
        child: Container(width: 36, height: 36,
          decoration: BoxDecoration(
            color: value < max ? const Color(0xFFFFF0F7) : Colors.grey.shade100,
            borderRadius: BorderRadius.circular(10)),
          child: Icon(Icons.add_rounded,
              color: value < max ? _pink : Colors.grey.shade300, size: 18))),
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
      Text('Loading cycle data…', style: TextStyle(color: _mid, fontSize: 14,
          decoration: TextDecoration.none)),
    ]));
} 