import 'package:flutter/material.dart';
import 'shehealth_dashboard.dart'; // Make sure this import is correct

class SymptomQuestionnaire extends StatefulWidget {
  const SymptomQuestionnaire({super.key});

  @override
  State<SymptomQuestionnaire> createState() => _SymptomQuestionnaireState();
}

class _SymptomQuestionnaireState extends State<SymptomQuestionnaire>
    with SingleTickerProviderStateMixin {
  int currentIndex = 0;
  Map<int, String> answers = {};
  late AnimationController _pulseController;
  bool _isProcessing = false;

  final List<Map<String, dynamic>> questions = [
    {'id': 1, 'question': "How's Your Cycle Regularity?", 'options': ["Regular", "Irregular", "Absent", "Unpredictable"]},
    {'id': 2, 'question': "Period Pain Level?", 'options': ["None", "Mild", "Moderate", "Severe"]},
    {'id': 3, 'question': "Any Weight Changes?", 'options': ["None", "Gain", "Loss", "Fluctuating"]},
    {'id': 4, 'question': "Fatigue Frequency?", 'options': ["Rarely", "Sometimes", "Often", "Always"]},
    {'id': 5, 'question': "Mood Swing Pattern?", 'options': ["None", "Occasional", "Frequent", "Severe"]},
    {'id': 6, 'question': "Hair Changes Noticed?", 'options': ["None", "Excessive Growth", "Hair Loss", "Both"]},
    {'id': 7, 'question': "Sleep Quality Rating?", 'options': ["Good", "Fair", "Poor", "Very Poor"]},
    {'id': 8, 'question': "Digestive Concerns?", 'options': ["None", "Bloating", "Constipation", "Severe"]},
    {'id': 9, 'question': "Skin Condition Status?", 'options': ["Clear", "Occasional Acne", "Persistent Acne", "Severe"]},
    {'id': 10, 'question': "Headache Frequency?", 'options': ["Rarely", "Monthly", "Weekly", "Daily"]}
  ];

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  void handleAnswer(String option) {
    setState(() {
      answers[questions[currentIndex]['id']] = option;
    });

    Future.delayed(const Duration(milliseconds: 300), () {
      if (currentIndex < questions.length - 1) {
        setState(() => currentIndex++);
      } else {
        setState(() => currentIndex++); // Trigger the "Completion" UI
        _completeAssessment();
      }
    });
  }

  void _completeAssessment() {
    setState(() => _isProcessing = true);
    Future.delayed(const Duration(seconds: 2), () {
      setState(() => _isProcessing = false);
    });
  }

  void _goToDashboard() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const DashboardPage()),
    );
  }

  double get progress => ((currentIndex) / questions.length).clamp(0.0, 1.0) * 100;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFFAF8F5), Color(0xFFFFF5F8)],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      if (currentIndex < questions.length) ...[
                        _buildQuestionCard(),
                        const SizedBox(height: 24),
                        _buildOptionsGrid(),
                        const SizedBox(height: 20),
                        _buildHelpfulTip(),
                      ] else
                        _buildCompletion(),
                    ],
                  ),
                ),
              ),
              _buildFooter(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFD4879C), Color(0xFFE5A1A1)],
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFD4879C).withOpacity(0.3),
            blurRadius: 15,
            offset: const Offset(0, 5),
          )
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.favorite,
                    color: Color(0xFFD4879C),
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Symptom Assessment',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (currentIndex < questions.length) ...[
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            const Icon(Icons.auto_awesome, color: Colors.white, size: 16),
                            const SizedBox(width: 6),
                            Text(
                              'Question ${currentIndex + 1} of 10',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 14,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: LinearProgressIndicator(
                value: progress / 100,
                backgroundColor: Colors.white.withOpacity(0.3),
                color: Colors.white,
                minHeight: 8,
              ),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: Text(
                '${progress.round()}% Complete',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuestionCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          )
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: const BoxDecoration(
              color: Color(0xFFE5A1A1),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.person,
              color: Colors.white,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'HEALTH ASSISTANT',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFFD4879C),
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  questions[currentIndex]['question'],
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2D2D2D),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOptionsGrid() {
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        childAspectRatio: 2.0,
        crossAxisSpacing: 16,
        mainAxisSpacing: 16,
      ),
      itemCount: questions[currentIndex]['options'].length,
      itemBuilder: (context, index) {
        final option = questions[currentIndex]['options'][index];
        return InkWell(
          onTap: () => handleAnswer(option),
          child: ClipPath(
            clipper: ParallelogramClipper(),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(
                  color: const Color(0xFFE5C4C4),
                  width: 2.5,
                ),
              ),
              child: Center(
                child: Text(
                  option,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 15,
                    color: Color(0xFF2D2D2D),
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildHelpfulTip() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF0F3),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFFE5C4C4).withOpacity(0.3),
          width: 1.5,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(
              Icons.auto_awesome,
              color: Color(0xFFD4879C),
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              "Tip: Answer based on your experiences over the past 3 months for more accurate insights.",
              style: TextStyle(
                fontSize: 13,
                color: Color(0xFF666666),
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCompletion() {
    return Column(
      children: [
        const Icon(Icons.favorite, size: 80, color: Color(0xFFD4879C)),
        const SizedBox(height: 16),
        const Text(
          "Assessment Complete!",
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: Color(0xFFD4879C),
          ),
        ),
        const SizedBox(height: 20),
        if (_isProcessing)
          const CircularProgressIndicator(
            color: Color(0xFFD4879C),
          )
        else
          ElevatedButton(
            onPressed: _goToDashboard,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFD4879C),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            child: const Text(
              "Go to Dashboard",
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildFooter() {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: const BoxDecoration(
              color: Color(0xFFD4879C),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.favorite,
              color: Colors.white,
              size: 16,
            ),
          ),
          const SizedBox(width: 10),
          const Text(
            "Backed By Medical Experts",
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: Color(0xFFD4879C),
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}

class ParallelogramClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    final path = Path();
    path.moveTo(size.width * 0.08, 0);
    path.lineTo(size.width, 0);
    path.lineTo(size.width * 0.92, size.height);
    path.lineTo(0, size.height);
    path.close();
    return path;
  }
  
  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) => false;
}