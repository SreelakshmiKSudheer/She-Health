import 'package:flutter/material.dart';

// ── Entry point ──────────────────────────────────────────────────────────────
class SurveyPage extends StatefulWidget {
  const SurveyPage({super.key});
  @override
  State<SurveyPage> createState() => _SurveyPageState();
}

// ── Color palette (matches app) ──────────────────────────────────────────────
const Color kPink      = Color(0xFFC85A7A);
const Color kPinkSoft  = Color(0xFFE59393);
const Color kPinkLight = Color(0xFFFFF5F8);
const Color kPinkBorder= Color(0xFFFCE7F3);
const Color kPinkDeep  = Color(0xFFFBCFE8);

// ── Survey catalogue ─────────────────────────────────────────────────────────
class SurveyMeta {
  final String id;
  final String title;
  final String subtitle;
  final IconData icon;
  final Color accent;
  final String duration;
  final String category;
  final List<SurveyQuestion> questions;

  const SurveyMeta({
    required this.id,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.accent,
    required this.duration,
    required this.category,
    required this.questions,
  });
}

class SurveyQuestion {
  final String question;
  final List<String> options;
  const SurveyQuestion({required this.question, required this.options});
}

final List<SurveyMeta> kSurveys = [
  SurveyMeta(
    id: 'menstrual_health',
    title: 'Menstrual Health',
    subtitle: 'Track your cycle patterns & symptoms',
    icon: Icons.water_drop,
    accent: kPink,
    duration: '3 min',
    category: 'Monthly',
    questions: [
      SurveyQuestion(question: 'How regular is your menstrual cycle?', options: ['Very Regular (28±2 days)', 'Somewhat Regular', 'Irregular', 'Very Irregular']),
      SurveyQuestion(question: 'How would you rate your period pain?', options: ['No Pain', 'Mild — manageable', 'Moderate — affects daily life', 'Severe — debilitating']),
      SurveyQuestion(question: 'How heavy is your flow on heaviest days?', options: ['Light', 'Moderate', 'Heavy', 'Very Heavy (clots)']),
      SurveyQuestion(question: 'Do you experience spotting between periods?', options: ['Never', 'Rarely', 'Sometimes', 'Often']),
      SurveyQuestion(question: 'How long does your period usually last?', options: ['1–3 days', '4–5 days', '6–7 days', 'More than 7 days']),
    ],
  ),
  SurveyMeta(
    id: 'pcos_screening',
    title: 'PCOS Screening',
    subtitle: 'Early detection of PCOS indicators',
    icon: Icons.health_and_safety,
    accent: Color(0xFFB5478A),
    duration: '4 min',
    category: 'Screening',
    questions: [
      SurveyQuestion(question: 'Have you noticed unexplained weight gain recently?', options: ['No', 'Slight gain (< 3 kg)', 'Moderate gain (3–8 kg)', 'Significant gain (> 8 kg)']),
      SurveyQuestion(question: 'Do you experience excessive hair growth (face, chest, back)?', options: ['Not at all', 'Slightly', 'Moderately', 'Significantly']),
      SurveyQuestion(question: 'How is your acne/skin condition?', options: ['Clear skin', 'Occasional breakouts', 'Frequent breakouts', 'Severe/persistent acne']),
      SurveyQuestion(question: 'Have you noticed hair thinning or loss on the scalp?', options: ['No', 'Slight thinning', 'Moderate thinning', 'Significant hair loss']),
      SurveyQuestion(question: 'Do you experience mood swings or depression?', options: ['Rarely', 'Occasionally', 'Frequently', 'Almost always']),
    ],
  ),
  SurveyMeta(
    id: 'thyroid_wellness',
    title: 'Thyroid Wellness',
    subtitle: 'Monitor thyroid health indicators',
    icon: Icons.monitor_heart,
    accent: Color(0xFF6B7FD4),
    duration: '3 min',
    category: 'Screening',
    questions: [
      SurveyQuestion(question: 'How are your energy levels throughout the day?', options: ['High energy', 'Moderate energy', 'Often tired', 'Constantly exhausted']),
      SurveyQuestion(question: 'How is your sensitivity to temperature?', options: ['Normal', 'Feel cold easily', 'Feel hot easily', 'Extreme sensitivity']),
      SurveyQuestion(question: 'Have you noticed changes in your weight without diet changes?', options: ['No change', 'Slight gain', 'Significant gain', 'Unexplained weight loss']),
      SurveyQuestion(question: 'Do you experience dry skin or hair?', options: ['No', 'Occasionally', 'Frequently', 'Severely']),
      SurveyQuestion(question: 'How is your heart rate on a typical day?', options: ['Normal', 'Occasionally fast', 'Often rapid', 'Slow/sluggish feeling']),
    ],
  ),
  SurveyMeta(
    id: 'mental_wellness',
    title: 'Mental Wellness',
    subtitle: 'Emotional health & stress assessment',
    icon: Icons.psychology,
    accent: Color(0xFF7C6FCD),
    duration: '4 min',
    category: 'Weekly',
    questions: [
      SurveyQuestion(question: 'How would you rate your stress levels this week?', options: ['Very low', 'Manageable', 'Quite stressed', 'Overwhelmed']),
      SurveyQuestion(question: 'How has your sleep quality been?', options: ['Excellent (7–9 hrs)', 'Good (6–7 hrs)', 'Poor (< 6 hrs)', 'Very poor / insomnia']),
      SurveyQuestion(question: 'How often have you felt anxious or worried?', options: ['Rarely', 'Occasionally', 'Frequently', 'Almost constantly']),
      SurveyQuestion(question: 'How would you describe your overall mood?', options: ['Happy & positive', 'Neutral', 'Often sad/low', 'Depressed']),
      SurveyQuestion(question: 'Do you feel supported by people around you?', options: ['Very supported', 'Mostly supported', 'Somewhat isolated', 'Very lonely']),
    ],
  ),
  SurveyMeta(
    id: 'nutrition_lifestyle',
    title: 'Nutrition & Lifestyle',
    subtitle: 'Diet, exercise & daily habits',
    icon: Icons.spa,
    accent: Color(0xFF2E8B57),
    duration: '3 min',
    category: 'Weekly',
    questions: [
      SurveyQuestion(question: 'How balanced is your daily diet?', options: ['Very balanced', 'Mostly balanced', 'Often unhealthy', 'Very poor diet']),
      SurveyQuestion(question: 'How much water do you drink daily?', options: ['> 2.5 L', '1.5–2.5 L', '0.5–1.5 L', '< 0.5 L']),
      SurveyQuestion(question: 'How often do you exercise per week?', options: ['5+ times', '3–4 times', '1–2 times', 'Rarely / never']),
      SurveyQuestion(question: 'How much processed / junk food do you consume?', options: ['Very rarely', 'Occasionally', 'Frequently', 'Daily']),
      SurveyQuestion(question: 'Do you take vitamins or supplements?', options: ['Yes, regularly', 'Sometimes', 'Rarely', 'Never']),
    ],
  ),
  SurveyMeta(
    id: 'bone_health',
    title: 'Bone & Joint Health',
    subtitle: 'Calcium, joints & bone density check',
    icon: Icons.accessibility_new,
    accent: Color(0xFFD4891A),
    duration: '3 min',
    category: 'Monthly',
    questions: [
      SurveyQuestion(question: 'Do you experience joint pain or stiffness?', options: ['Never', 'Occasionally', 'Frequently', 'Daily']),
      SurveyQuestion(question: 'How much dairy / calcium-rich food do you consume?', options: ['Daily', 'A few times/week', 'Rarely', 'Never']),
      SurveyQuestion(question: 'Do you get adequate sunlight for Vitamin D?', options: ['Yes, daily', 'A few times/week', 'Rarely', 'Almost never']),
      SurveyQuestion(question: 'Do you have a family history of osteoporosis?', options: ['No known history', 'Possibly', 'Yes', 'Not sure']),
      SurveyQuestion(question: 'Do you experience back pain or frequent fractures?', options: ['Never', 'Occasionally', 'Frequently', 'Ongoing issues']),
    ],
  ),
];

// ── Survey List Page ──────────────────────────────────────────────────────────
class _SurveyPageState extends State<SurveyPage> {
  String _selectedCategory = 'All';
  final List<String> _categories = ['All', 'Monthly', 'Weekly', 'Screening'];
  final Map<String, bool> _completedSurveys = {};

  List<SurveyMeta> get _filtered => _selectedCategory == 'All'
      ? kSurveys
      : kSurveys.where((s) => s.category == _selectedCategory).toList();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kPinkLight,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildCategoryFilter(),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                itemCount: _filtered.length,
                itemBuilder: (ctx, i) => _buildSurveyCard(_filtered[i]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Header ──────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    final completed = _completedSurveys.values.where((v) => v).length;
    return Stack(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 28),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [kPink, kPinkSoft],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back, color: Colors.white, size: 26),
                    padding: EdgeInsets.zero,
                  ),
                  const SizedBox(width: 8),
                  const Icon(Icons.assignment, color: Colors.white, size: 32),
                  const Spacer(),
                  // Progress badge
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.20),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.white, size: 16),
                        const SizedBox(width: 6),
                        Text(
                          '$completed / ${kSurveys.length} done',
                          style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w700),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              const Text(
                'Health Surveys',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.5),
              ),
              const SizedBox(height: 4),
              const Text(
                'Track & monitor your health with targeted surveys',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 18),
              // Overall progress bar
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Overall Completion',
                          style: TextStyle(color: Colors.white70, fontSize: 12)),
                      Text('${(completed / kSurveys.length * 100).round()}%',
                          style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: LinearProgressIndicator(
                      value: completed / kSurveys.length,
                      backgroundColor: Colors.white.withOpacity(0.25),
                      valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                      minHeight: 8,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        // Decorative circles
        Positioned(
          top: -30, right: -30,
          child: Container(
            width: 120, height: 120,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.08), shape: BoxShape.circle),
          ),
        ),
        Positioned(
          bottom: 0, left: -20,
          child: Container(
            width: 70, height: 70,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.08), shape: BoxShape.circle),
          ),
        ),
      ],
    );
  }

  // ── Category Filter ─────────────────────────────────────────────────────────
  Widget _buildCategoryFilter() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: _categories.map((cat) {
          final active = _selectedCategory == cat;
          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: GestureDetector(
              onTap: () => setState(() => _selectedCategory = cat),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  gradient: active
                      ? const LinearGradient(colors: [kPink, kPinkSoft])
                      : null,
                  color: active ? null : kPinkLight,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: active ? kPink : kPinkBorder, width: 1.5),
                ),
                child: Text(
                  cat,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: active ? Colors.white : kPink.withOpacity(0.7),
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Survey Card ─────────────────────────────────────────────────────────────
  Widget _buildSurveyCard(SurveyMeta survey) {
    final done = _completedSurveys[survey.id] ?? false;
    return GestureDetector(
      onTap: () async {
        final result = await Navigator.push<bool>(
          context,
          MaterialPageRoute(
            builder: (_) => SurveyDetailPage(survey: survey),
          ),
        );
        if (result == true) {
          setState(() => _completedSurveys[survey.id] = true);
        }
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
              color: done ? survey.accent.withOpacity(0.3) : kPinkBorder,
              width: 1.5),
          boxShadow: [
            BoxShadow(
                color: kPink.withOpacity(0.07),
                blurRadius: 12,
                offset: const Offset(0, 4)),
          ],
        ),
        child: Column(
          children: [
            // Top accent bar
            Container(
              height: 4,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                    colors: [survey.accent, survey.accent.withOpacity(0.4)]),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(18)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  // Icon box
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: survey.accent.withOpacity(0.10),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Icon(survey.icon, color: survey.accent, size: 26),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                survey.title,
                                style: const TextStyle(
                                    fontSize: 15,
                                    fontWeight: FontWeight.w800,
                                    color: Colors.black87),
                              ),
                            ),
                            if (done)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: Colors.green.shade50,
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(
                                      color: Colors.green.shade200),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.check_circle,
                                        color: Colors.green.shade600, size: 12),
                                    const SizedBox(width: 3),
                                    Text('Done',
                                        style: TextStyle(
                                            color: Colors.green.shade600,
                                            fontSize: 10,
                                            fontWeight: FontWeight.bold)),
                                  ],
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 3),
                        Text(
                          survey.subtitle,
                          style: TextStyle(
                              fontSize: 12, color: Colors.grey.shade600),
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            _buildChip(Icons.timer_outlined, survey.duration,
                                kPinkBorder, kPink),
                            const SizedBox(width: 8),
                            _buildChip(Icons.category_outlined, survey.category,
                                survey.accent.withOpacity(0.12),
                                survey.accent),
                            const SizedBox(width: 8),
                            _buildChip(Icons.quiz_outlined,
                                '${survey.questions.length} Qs',
                                kPinkBorder, kPink),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Arrow
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: done
                          ? Colors.green.shade50
                          : kPinkBorder,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      done ? Icons.refresh : Icons.arrow_forward_ios,
                      color: done ? Colors.green.shade600 : kPink,
                      size: 14,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChip(IconData icon, String label, Color bg, Color fg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
          color: bg, borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 11, color: fg),
          const SizedBox(width: 4),
          Text(label,
              style: TextStyle(
                  fontSize: 10, color: fg, fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

// ── Survey Detail / Question Page ────────────────────────────────────────────
class SurveyDetailPage extends StatefulWidget {
  final SurveyMeta survey;
  const SurveyDetailPage({super.key, required this.survey});
  @override
  State<SurveyDetailPage> createState() => _SurveyDetailPageState();
}

class _SurveyDetailPageState extends State<SurveyDetailPage>
    with SingleTickerProviderStateMixin {
  int _currentIndex = 0;
  final Map<int, int> _answers = {};
  bool _submitted = false;

  late AnimationController _animCtrl;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);
    _animCtrl.forward();
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    super.dispose();
  }

  void _nextQuestion() {
    if (_currentIndex < widget.survey.questions.length - 1) {
      _animCtrl.reverse().then((_) {
        setState(() => _currentIndex++);
        _animCtrl.forward();
      });
    } else {
      setState(() => _submitted = true);
    }
  }

  void _prevQuestion() {
    if (_currentIndex > 0) {
      _animCtrl.reverse().then((_) {
        setState(() => _currentIndex--);
        _animCtrl.forward();
      });
    }
  }

  double get _progress =>
      (_currentIndex + 1) / widget.survey.questions.length;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kPinkLight,
      body: SafeArea(
        child: _submitted ? _buildResultPage() : _buildQuestionPage(),
      ),
    );
  }

  // ── Question Page ───────────────────────────────────────────────────────────
  Widget _buildQuestionPage() {
    final q = widget.survey.questions[_currentIndex];
    final accent = widget.survey.accent;

    return Column(
      children: [
        // Header
        Container(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
                colors: [accent, accent.withOpacity(0.7)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close, color: Colors.white, size: 24),
                    padding: EdgeInsets.zero,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.survey.title,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 17,
                          fontWeight: FontWeight.bold),
                    ),
                  ),
                  Text(
                    '${_currentIndex + 1} / ${widget.survey.questions.length}',
                    style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                        fontWeight: FontWeight.w600),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              // Progress bar
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: LinearProgressIndicator(
                  value: _progress,
                  backgroundColor: Colors.white.withOpacity(0.25),
                  valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                  minHeight: 6,
                ),
              ),
            ],
          ),
        ),

        Expanded(
          child: FadeTransition(
            opacity: _fadeAnim,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Question number badge
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: accent.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      'Question ${_currentIndex + 1}',
                      style: TextStyle(
                          fontSize: 12,
                          color: accent,
                          fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(height: 14),
                  // Question text
                  Text(
                    q.question,
                    style: const TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                        color: Colors.black87,
                        height: 1.4),
                  ),
                  const SizedBox(height: 24),
                  // Options
                  ...List.generate(q.options.length, (i) {
                    final selected = _answers[_currentIndex] == i;
                    return GestureDetector(
                      onTap: () =>
                          setState(() => _answers[_currentIndex] = i),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: selected
                              ? accent.withOpacity(0.08)
                              : Colors.white,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: selected ? accent : kPinkBorder,
                            width: selected ? 2 : 1.5,
                          ),
                          boxShadow: selected
                              ? [
                                  BoxShadow(
                                      color: accent.withOpacity(0.15),
                                      blurRadius: 10,
                                      offset: const Offset(0, 3))
                                ]
                              : [
                                  BoxShadow(
                                      color: kPink.withOpacity(0.05),
                                      blurRadius: 6,
                                      offset: const Offset(0, 2))
                                ],
                        ),
                        child: Row(
                          children: [
                            // Circle indicator
                            Container(
                              width: 22,
                              height: 22,
                              decoration: BoxDecoration(
                                color: selected ? accent : Colors.transparent,
                                shape: BoxShape.circle,
                                border: Border.all(
                                    color: selected ? accent : Colors.grey.shade300,
                                    width: 2),
                              ),
                              child: selected
                                  ? const Icon(Icons.check,
                                      color: Colors.white, size: 14)
                                  : null,
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Text(
                                q.options[i],
                                style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: selected
                                        ? FontWeight.w700
                                        : FontWeight.w500,
                                    color: selected
                                        ? accent
                                        : Colors.black87),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }),
                ],
              ),
            ),
          ),
        ),

        // Bottom navigation
        Container(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border(top: BorderSide(color: kPinkBorder, width: 1.5)),
          ),
          child: Row(
            children: [
              if (_currentIndex > 0)
                OutlinedButton.icon(
                  onPressed: _prevQuestion,
                  icon: const Icon(Icons.arrow_back_ios, size: 14),
                  label: const Text('Back'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: kPink,
                    side: const BorderSide(color: kPinkBorder, width: 1.5),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              const Spacer(),
              ElevatedButton(
                onPressed: _answers.containsKey(_currentIndex)
                    ? _nextQuestion
                    : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: accent,
                  disabledBackgroundColor: kPinkBorder,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 0,
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _currentIndex == widget.survey.questions.length - 1
                          ? 'Submit'
                          : 'Next',
                      style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 15),
                    ),
                    const SizedBox(width: 6),
                    Icon(
                      _currentIndex == widget.survey.questions.length - 1
                          ? Icons.check
                          : Icons.arrow_forward_ios,
                      color: Colors.white,
                      size: 14,
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

  // ── Result Page ─────────────────────────────────────────────────────────────
  Widget _buildResultPage() {
    final accent = widget.survey.accent;
    final answeredAll =
        _answers.length == widget.survey.questions.length;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const SizedBox(height: 20),
          // Success icon
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                  colors: [accent, accent.withOpacity(0.6)]),
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                    color: accent.withOpacity(0.30),
                    blurRadius: 24,
                    offset: const Offset(0, 8)),
              ],
            ),
            child: const Icon(Icons.check_circle_outline,
                color: Colors.white, size: 52),
          ),
          const SizedBox(height: 24),
          Text(
            'Survey Complete! 🎉',
            style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w900,
                color: accent),
          ),
          const SizedBox(height: 8),
          Text(
            widget.survey.title,
            style: const TextStyle(
                fontSize: 16,
                color: Colors.black54,
                fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 28),

          // Summary card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: kPinkBorder, width: 1.5),
              boxShadow: [
                BoxShadow(
                    color: kPink.withOpacity(0.08),
                    blurRadius: 14,
                    offset: const Offset(0, 4)),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                          color: kPinkBorder,
                          borderRadius: BorderRadius.circular(10)),
                      child: const Icon(Icons.summarize_outlined,
                          color: kPink, size: 20),
                    ),
                    const SizedBox(width: 10),
                    const Text('Your Responses',
                        style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.black87)),
                  ],
                ),
                const SizedBox(height: 16),
                const Divider(color: kPinkBorder),
                const SizedBox(height: 12),
                ...List.generate(widget.survey.questions.length, (i) {
                  final ans = _answers[i];
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Q${i + 1}. ${widget.survey.questions[i].question}',
                          style: const TextStyle(
                              fontSize: 12,
                              color: Colors.black54,
                              fontWeight: FontWeight.w500),
                        ),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            color: accent.withOpacity(0.08),
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(
                                color: accent.withOpacity(0.25)),
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.check_circle,
                                  color: accent, size: 14),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  ans != null
                                      ? widget.survey.questions[i].options[ans]
                                      : 'Not answered',
                                  style: TextStyle(
                                      fontSize: 13,
                                      color: ans != null
                                          ? Colors.black87
                                          : Colors.grey,
                                      fontWeight: FontWeight.w600),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  );
                }),
              ],
            ),
          ),

          const SizedBox(height: 20),

          // Info note
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: kPinkBorder,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: kPinkDeep),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.info_outline, color: kPink, size: 18),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Your responses have been recorded. This data helps generate a more accurate health report. Consult a healthcare professional for medical advice.',
                    style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade700,
                        height: 1.5),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // Done button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () => Navigator.pop(context, true),
              icon: const Icon(Icons.home, color: Colors.white),
              label: const Text(
                'Back to Surveys',
                style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: kPink,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
                elevation: 0,
              ),
            ),
          ),
          const SizedBox(height: 12),
          // Retake button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () {
                setState(() {
                  _currentIndex = 0;
                  _answers.clear();
                  _submitted = false;
                });
                _animCtrl.forward();
              },
              icon: const Icon(Icons.refresh, color: kPink),
              label: const Text(
                'Retake Survey',
                style: TextStyle(
                    color: kPink,
                    fontWeight: FontWeight.bold,
                    fontSize: 15),
              ),
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: kPinkBorder, width: 1.5),
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
            ),
          ),
          const SizedBox(height: 30),
        ],
      ),
    );
  }
}