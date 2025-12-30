import 'package:flutter/material.dart';

// -----------------------------------------------------------------------------
// The Detailed Health Report Page (Originally First Code Block)
// -----------------------------------------------------------------------------

class HealthReportPage extends StatefulWidget {
  const HealthReportPage({super.key});

  @override
  State<HealthReportPage> createState() => _HealthReportPageState();
}

class _HealthReportPageState extends State<HealthReportPage> {
  final Color pinkStart = const Color(0xFFC85A7A);
  final Color pinkEnd = const Color(0xFFE59393);

  // Sample data - replace with actual user data
  final Map<String, dynamic> reportData = {
    'patientName': 'Sarah Anderson',
    'patientId': 'SH2024001',
    'date': 'October 12, 2025',
    'age': '28 years',
    'assessmentDate': 'October 10, 2025',

    'symptoms': [
      {'name': 'Cycle Regularity', 'value': 'Regular'},
      {'name': 'Period Pain Level', 'value': 'Mild'},
      {'name': 'Weight Changes', 'value': 'None'},
      {'name': 'Fatigue Frequency', 'value': 'Sometimes'},
      {'name': 'Mood Swing Pattern', 'value': 'Occasional'},
    ],

    'riskAssessment': [
      {'condition': 'PCOD/PCOS', 'risk': 'Low', 'percentage': '15%', 'color': Colors.green},
      {'condition': 'Thyroid Disorders', 'risk': 'No Risk', 'percentage': '5%', 'color': Colors.green},
      {'condition': 'Endometriosis', 'risk': 'Monitor', 'percentage': '35%', 'color': Colors.orange},
      {'condition': 'Cervical Cancer', 'risk': 'No Risk', 'percentage': '3%', 'color': Colors.green},
    ],

    'recommendations': [
      'Maintain regular menstrual cycle tracking',
      'Continue balanced diet with iron-rich foods',
      'Practice stress management techniques',
      'Schedule regular gynecological check-ups',
      'Monitor symptoms and report any changes',
    ],

    'lifestyle': {
      'exercise': 'Moderate (3-4 times/week)',
      'sleep': '7-8 hours',
      'water': '1.5-2L daily',
      'stress': 'Moderate',
    }
  };

  void _downloadPDF() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('PDF download functionality will be added'),
        backgroundColor: Color(0xFFC85A7A),
      ),
    );
  }

  void _shareReport() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Share report functionality will be added'),
        backgroundColor: Color(0xFFC85A7A),
      ),
    );
  }

  void _emailReport() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Email report functionality will be added'),
        backgroundColor: Color(0xFFC85A7A),
      ),
    );
  }

  void _printReport() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Print report functionality will be added'),
        backgroundColor: Color(0xFFC85A7A),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFFF5F8),
      body: SafeArea(
        child: Column(
          children: [
            // HEADER
            Stack(
              children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [pinkStart, pinkEnd],
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
                            icon: const Icon(Icons.arrow_back, color: Colors.white, size: 28),
                          ),
                          const SizedBox(width: 10),
                          const Icon(Icons.description, color: Colors.white, size: 40),
                        ],
                      ),
                      const SizedBox(height: 10),
                      const Text(
                        "Health Report",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1,
                        ),
                      ),
                      const Text(
                        "Your comprehensive health assessment",
                        style: TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                    ],
                  ),
                ),
                // Decorative circles
                Positioned(
                  top: -40,
                  right: -40,
                  child: Container(
                    width: 150,
                    height: 150,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.12),
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ],
            ),

            // MAIN CONTENT
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Patient Info Card
                    _buildSectionCard(
                      'Patient Information',
                      Icons.person,
                      Column(
                        children: [
                          _buildInfoItem('Name', reportData['patientName']),
                          _buildInfoItem('Patient ID', reportData['patientId']),
                          _buildInfoItem('Age', reportData['age']),
                          _buildInfoItem('Report Date', reportData['date']),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Symptom Summary Card
                    _buildSectionCard(
                      'Symptom Summary',
                      Icons.favorite,
                      Column(
                        children: (reportData['symptoms'] as List).map((symptom) {
                          return _buildSymptomItem(
                            symptom['name'],
                            symptom['value'],
                          );
                        }).toList(),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Risk Assessment Card
                    _buildSectionCard(
                      'Risk Assessment',
                      Icons.shield,
                      Column(
                        children: (reportData['riskAssessment'] as List).map((risk) {
                          return _buildRiskItem(
                            risk['condition'],
                            risk['risk'],
                            risk['percentage'],
                            risk['color'],
                          );
                        }).toList(),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Lifestyle Factors Card
                    _buildSectionCard(
                      'Lifestyle Factors',
                      Icons.accessibility_new,
                      Column(
                        children: [
                          _buildLifestyleItem(Icons.fitness_center, 'Exercise', reportData['lifestyle']['exercise']),
                          _buildLifestyleItem(Icons.bedtime, 'Sleep', reportData['lifestyle']['sleep']),
                          _buildLifestyleItem(Icons.water_drop, 'Water Intake', reportData['lifestyle']['water']),
                          _buildLifestyleItem(Icons.psychology, 'Stress Level', reportData['lifestyle']['stress']),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Recommendations Card
                    _buildSectionCard(
                      'Health Recommendations',
                      Icons.lightbulb,
                      Column(
                        children: (reportData['recommendations'] as List).map((rec) {
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Container(
                                  margin: const EdgeInsets.only(top: 6),
                                  width: 6,
                                  height: 6,
                                  decoration: BoxDecoration(
                                    color: pinkStart,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    rec,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      color: Colors.black87,
                                      height: 1.5,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          );
                        }).toList(),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Disclaimer Card
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFCE7F3),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFFFBCFE8), width: 2),
                      ),
                      child: Column(
                        children: [
                          Row(
                            children: [
                              Icon(Icons.info_outline, color: pinkStart, size: 20),
                              const SizedBox(width: 8),
                              const Text(
                                'Disclaimer',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 14,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          const Text(
                            'This report provides risk predictions and preventive advice based on self-reported data. It is not a medical diagnosis. Please consult with a healthcare professional for proper medical evaluation.',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.black87,
                              height: 1.5,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 24),

                    // Action Buttons Grid
                    GridView.count(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      crossAxisCount: 2,
                      crossAxisSpacing: 12,
                      mainAxisSpacing: 12,
                      childAspectRatio: 2.5,
                      children: [
                        _buildActionButton(
                          icon: Icons.picture_as_pdf,
                          label: 'Download PDF',
                          onTap: _downloadPDF,
                        ),
                        _buildActionButton(
                          icon: Icons.share,
                          label: 'Share Report',
                          onTap: _shareReport,
                        ),
                        _buildActionButton(
                          icon: Icons.email,
                          label: 'Email Report',
                          onTap: _emailReport,
                        ),
                        _buildActionButton(
                          icon: Icons.print,
                          label: 'Print Report',
                          onTap: _printReport,
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),

                    // Back to Dashboard Button
                    OutlinedButton.icon(
                      onPressed: () => Navigator.pop(context),
                      icon: Icon(Icons.home, color: pinkStart),
                      label: Text(
                        'Back to Dashboard',
                        style: TextStyle(
                          color: pinkStart,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: pinkStart, width: 2),
                        minimumSize: const Size(double.infinity, 54),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // --- Widget Builders for HealthReportPage ---

  Widget _buildSectionCard(String title, IconData icon, Widget content) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 2),
        boxShadow: [
          BoxShadow(
            color: Colors.pink.shade100.withOpacity(0.3),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFCE7F3),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: pinkStart, size: 24),
              ),
              const SizedBox(width: 12),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          const Divider(color: Color(0xFFFCE7F3), thickness: 1),
          const SizedBox(height: 16),
          content,
        ],
      ),
    );
  }

  Widget _buildInfoItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.grey,
              fontWeight: FontWeight.w500,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.black87,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSymptomItem(String name, String value) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF5F8),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            name,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.black87,
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: pinkStart.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: pinkStart.withOpacity(0.3)),
            ),
            child: Text(
              value,
              style: TextStyle(
                fontSize: 13,
                color: pinkStart,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskItem(String condition, String risk, String percentage, Color color) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.3), width: 1.5),
      ),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: Text(
              condition,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
            ),
          ),
          Expanded(
            flex: 2,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                risk,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            percentage,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLifestyleItem(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: pinkStart.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: pinkStart, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                color: Colors.grey,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.black87,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [pinkStart, pinkEnd],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: pinkStart.withOpacity(0.3),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 28),
            const SizedBox(height: 6),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// The Main Application and Questionnaire Logic (Originally Second Code Block)
// -----------------------------------------------------------------------------

void main() {
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: SymptomQuestionnaire(),
  ));
}

// Placeholder ReportPage is removed. We navigate directly to HealthReportPage.

// MAIN QUESTIONNAIRE
class SymptomQuestionnaire extends StatefulWidget {
  const SymptomQuestionnaire({super.key});

  @override
  State<SymptomQuestionnaire> createState() => _SymptomQuestionnaireState();
}

class _SymptomQuestionnaireState extends State<SymptomQuestionnaire> {
  final Color pinkStart = const Color(0xFFC85A7A);
  final Color pinkEnd = const Color(0xFFE59393);

  int currentIndex = 0;
  final Map<int, String> answers = {};

  final List<Map<String, dynamic>> questions = [
    {
      'id': 1,
      'question': "How's Your Cycle Regularity?",
      'options': ["Regular", "Irregular", "Absent", "Unpredictable"]
    },
    {
      'id': 2,
      'question': "Period Pain Level?",
      'options': ["None", "Mild", "Moderate", "Severe"]
    },
    {
      'id': 3,
      'question': "Any Weight Changes?",
      'options': ["None", "Gain", "Loss", "Fluctuating"]
    },
    {
      'id': 4,
      'question': "Fatigue Frequency?",
      'options': ["Rarely", "Sometimes", "Often", "Always"]
    },
    {
      'id': 5,
      'question': "Mood Swing Pattern?",
      'options': ["None", "Occasional", "Frequent", "Severe"]
    },
  ];

  void handleAnswer(String option) {
    final qid = questions[currentIndex]['id'] as int;
    setState(() {
      answers[qid] = option;
    });
  }

  void goNext() {
    // Only proceed if an answer is selected
    if (answers.containsKey(questions[currentIndex]['id'] as int)) {
      if (currentIndex < questions.length - 1) {
        setState(() {
          currentIndex++;
        });
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an option to continue.'),
          backgroundColor: Color(0xFFC85A7A),
        ),
      );
    }
  }

  // MODIFIED: Navigates to the detailed HealthReportPage
  void goToReportPage() {
    // Before navigating, ensure the final question is answered
    if (answers.containsKey(questions.last['id'] as int)) {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const HealthReportPage()),
      );
    } else {
       ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an option to generate the report.'),
          backgroundColor: Color(0xFFC85A7A),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final q = questions[currentIndex];
    final opts = q['options'] as List<String>;
    final id = q['id'] as int;
    final progress = ((currentIndex + 1) / questions.length) * 100;

    return Scaffold(
      backgroundColor: const Color(0xFFFFF5F8),
      body: SafeArea(
        child: Column(
          children: [
            // HEADER
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [pinkStart, pinkEnd],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.favorite, color: Colors.white, size: 40),
                  const SizedBox(height: 10),
                  const Text(
                    "Symptom Assessment",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    "Question ${currentIndex + 1} of ${questions.length}",
                    style: const TextStyle(color: Colors.white70),
                  ),
                  const SizedBox(height: 10),
                  // Progress bar
                  Stack(
                    children: [
                      Container(
                        height: 6,
                        decoration: BoxDecoration(
                          color: Colors.white30,
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 400),
                        height: 6,
                        width:
                            MediaQuery.of(context).size.width * (progress / 100),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Colors.white, Colors.white70],
                          ),
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // MAIN CONTENT
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    // QUESTION CARD
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(18),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.pink.shade100.withOpacity(0.4),
                            blurRadius: 10,
                            offset: const Offset(2, 5),
                          )
                        ],
                      ),
                      child: Text(
                        q['question'] as String,
                        style: const TextStyle(
                          fontSize: 18,
                          color: Colors.black87,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),

                    const SizedBox(height: 25),

                    // OPTIONS GRID
                    GridView.builder(
                      itemCount: opts.length,
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        crossAxisSpacing: 12,
                        mainAxisSpacing: 12,
                        childAspectRatio: 2.6,
                      ),
                      itemBuilder: (context, index) {
                        final option = opts[index];
                        final bool selected = answers[id] == option;

                        return GestureDetector(
                          onTap: () => handleAnswer(option),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            child: CustomPaint(
                              painter: DownwardTailPainter(
                                color: selected ? pinkStart : Colors.white,
                                borderColor: selected
                                    ? pinkStart
                                    : const Color(0xFFE5C4C4),
                              ),
                              child: Container(
                                alignment: Alignment.center,
                                padding:
                                    const EdgeInsets.symmetric(horizontal: 10),
                                child: Text(
                                  option,
                                  style: TextStyle(
                                    color: selected
                                        ? Colors.white
                                        : Colors.black87,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),

                    const SizedBox(height: 30),

                    // BUTTONS
                    if (currentIndex < questions.length - 1)
                      ElevatedButton(
                        onPressed: goNext,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: pinkStart,
                          minimumSize: const Size(double.infinity, 48),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                        child: const Text(
                          "Next",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      )
                    else
                      // Final button navigates to the detailed HealthReportPage
                      ElevatedButton(
                        onPressed: goToReportPage,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: pinkStart,
                          minimumSize: const Size(double.infinity, 48),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                        child: const Text(
                          "Generate Report",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 🎨 Custom Painter for Parallelogram with Downward Slanted Tail
class DownwardTailPainter extends CustomPainter {
  final Color color;
  final Color borderColor;

  DownwardTailPainter({required this.color, required this.borderColor});

  @override
  void paint(Canvas canvas, Size size) {
    const double tailWidth = 12;
    const double tailHeight = 10;
    const double slant = 10;

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final borderPaint = Paint()
      ..color = borderColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final path = Path();
    path.moveTo(slant, 0);
    path.lineTo(size.width - slant, 0);
    path.lineTo(size.width, size.height - tailHeight - 5);
    path.lineTo(size.width - tailWidth, size.height - tailHeight + 3);
    path.lineTo(size.width - 18, size.height);
    path.lineTo(slant, size.height);
    path.lineTo(0, tailHeight);
    path.close();

    canvas.drawShadow(path, Colors.black38, 3, false);
    canvas.drawPath(path, paint);
    canvas.drawPath(path, borderPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}