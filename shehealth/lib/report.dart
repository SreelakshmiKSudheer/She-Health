import 'package:flutter/material.dart';
import 'models/app_models.dart';
import 'services/backend_api_service.dart';
import 'services/local_storage_service.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

class HealthReportPage extends StatefulWidget {
  final String? userId;
  final Map<String, dynamic>? predictionData;
  final LocalUserProfile? localUser;
  final String? reportText;
  const HealthReportPage({
    super.key,
    this.userId,
    this.predictionData,
    this.localUser,
    this.reportText,
  });

  @override
  State<HealthReportPage> createState() => _HealthReportPageState();
}

class _HealthReportPageState extends State<HealthReportPage>
    with SingleTickerProviderStateMixin {
  final Color pinkStart = const Color(0xFFC85A7A);
  final Color pinkEnd = const Color(0xFFE59393);

  late AnimationController _animController;
  late Animation<double> _barAnimation;

  // Risk config ΓÇö all pink palette, intensity shows severity
  static Map<String, dynamic> riskConfig(double p) {
    if (p < 0.10) {
      return {
        'label': 'No Risk',
        'color': const Color(0xFF9D8EC7), // soft lavender
        'bg': const Color(0xFFF3F0FB),
        'border': const Color(0xFFD4CCF0),
        'icon': Icons.check_circle_outline,
        'intensity': 0,
      };
    } else if (p < 0.30) {
      return {
        'label': 'Low Risk',
        'color': const Color(0xFFC85A7A), // app pink primary
        'bg': const Color(0xFFFFF5F8),
        'border': const Color(0xFFFCE7F3),
        'icon': Icons.thumb_up_outlined,
        'intensity': 1,
      };
    } else if (p < 0.55) {
      return {
        'label': 'Moderate Risk',
        'color': const Color(0xFFB8436A), // medium deep pink
        'bg': const Color(0xFFFEECF3),
        'border': const Color(0xFFF9B8D0),
        'icon': Icons.warning_amber_outlined,
        'intensity': 2,
      };
    } else if (p < 0.75) {
      return {
        'label': 'High Risk',
        'color': const Color(0xFF9E2D57), // deep rose
        'bg': const Color(0xFFFCE4EE),
        'border': const Color(0xFFF4A0BF),
        'icon': Icons.error_outline,
        'intensity': 3,
      };
    } else {
      return {
        'label': 'Very High Risk',
        'color': const Color(0xFF7B1D3F), // dark crimson rose
        'bg': const Color(0xFFF9D6E4),
        'border': const Color(0xFFEC87AD),
        'icon': Icons.dangerous_outlined,
        'intensity': 4,
      };
    }
  }

  String _resolveAge() {
    final predictionAge = (widget.predictionData ?? _fetchedPrediction)?['age'];
    if (predictionAge is num && predictionAge > 0) {
      return predictionAge.toInt().toString();
    }

    final profile = (widget.predictionData ?? _fetchedPrediction)?['user_profile'];
    if (profile is Map<String, dynamic>) {
      final profileAge = profile['age'];
      if (profileAge is num && profileAge > 0) {
        return profileAge.toInt().toString();
      }
    }

    final dob = (_localUserState ?? widget.localUser)?.dob;
    if (dob != null && dob.isNotEmpty) {
      final calculated = _calculateAgeFromDob(dob);
      if (calculated != null && calculated > 0) {
        return calculated.toString();
      }
    }

    return 'Not available';
  }

  int? _calculateAgeFromDob(String dob) {
    try {
      final parts = dob.split('/');
      if (parts.length != 3) {
        return null;
      }

      final day = int.parse(parts[0]);
      final month = int.parse(parts[1]);
      final year = int.parse(parts[2]);
      final birthDate = DateTime(year, month, day);
      final now = DateTime.now();

      var age = now.year - birthDate.year;
      final hasBirthdayPassed = now.month > birthDate.month ||
          (now.month == birthDate.month && now.day >= birthDate.day);
      if (!hasBirthdayPassed) {
        age -= 1;
      }

      return age;
    } catch (_) {
      return null;
    }
  }

  Map<String, dynamic> get reportData {
    final rawPredictions = (widget.predictionData ?? _fetchedPrediction)?['predictions'];
    List<Map<String, dynamic>> riskAssessment = [];

    if (rawPredictions is Map<String, dynamic>) {
      riskAssessment = rawPredictions.entries.map((entry) {
        final value = entry.value is Map<String, dynamic>
            ? entry.value as Map<String, dynamic>
            : <String, dynamic>{};
        final probabilityPct = (value['probability'] as num? ?? 0).toDouble();
        return {
          'condition': entry.key,
          'probability': (probabilityPct / 100).clamp(0.0, 1.0),
        };
      }).toList();
    }

    if (riskAssessment.isEmpty) {
      riskAssessment = [
        {'condition': 'PCOD / PCOS', 'probability': 0.15},
        {'condition': 'Thyroid Disorders', 'probability': 0.05},
        {'condition': 'Endometriosis', 'probability': 0.42},
        {'condition': 'Cervical Cancer', 'probability': 0.03},
      ];
    }

    riskAssessment.sort(
      (a, b) => ((b['probability'] as num?) ?? 0)
          .toDouble()
          .compareTo(((a['probability'] as num?) ?? 0).toDouble()),
    );

    final now = DateTime.now();
    final date =
        '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';

    return {
      'patientName': (_localUserState ?? widget.localUser)?.fullName ?? 'Unknown',
      'patientId': widget.userId ?? (_localUserState ?? widget.localUser)?.userId ?? 'Not available',
      'date': date,
      'age': _resolveAge(),
      'assessmentDate': date,
      'symptoms': [
        {
          'name': 'Blood Group',
          'value': (_localUserState ?? widget.localUser)?.bloodGroup ?? 'Not provided'
        },
        {
          'name': 'Marital Status',
          'value': (_localUserState ?? widget.localUser)?.maritalStatus ?? 'Not provided'
        },
        {
          'name': 'Activity Level',
          'value': (_localUserState ?? widget.localUser)?.activityLevel ?? 'Not provided'
        },
      ],
      'riskAssessment': riskAssessment,
      'recommendations': [
        'Maintain regular cycle and symptom tracking.',
        'Follow a balanced diet, hydration, and sleep routine.',
        'Consult a healthcare professional for clinical guidance.',
      ],
      'lifestyle': {
        'exercise': (_localUserState ?? widget.localUser)?.activityLevel ?? 'Not provided',
        'sleep': 'Track your sleep quality daily',
        'water': 'Aim for 1.5-2L daily',
        'stress': 'Practice stress-management routines',
      },
    };
  }

  // Overall risk = highest probability among all conditions
  double get _overallProbability {
    final risks = reportData['riskAssessment'] as List;
    return risks
        .map((r) => (r['probability'] as num).toDouble())
        .reduce((a, b) => a > b ? a : b);
  }

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );
    _barAnimation = CurvedAnimation(
      parent: _animController,
      curve: Curves.easeOutCubic,
    );
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _animController.forward();
    });
    // If no prediction data was provided, fetch latest from backend
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _fetchLatestPredictionIfNeeded();
    });
  }

  final BackendApiService _api = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;
  Map<String, dynamic>? _fetchedPrediction;
  bool _isFetchingPrediction = false;
  LocalUserProfile? _localUserState;

  Future<void> _fetchLatestPredictionIfNeeded() async {
    if (widget.predictionData != null) return;
    if (widget.userId == null || widget.userId!.isEmpty) return;
    if (_isFetchingPrediction) return;
    _isFetchingPrediction = true;
    try {
      final latest = await _api.getLatestPrediction(widget.userId!);
      final local = await _localStorage.findByUserId(widget.userId!);
      if (!mounted) return;
      setState(() {
        _fetchedPrediction = latest;
        if (widget.localUser == null) {
          _localUserState = local;
        }
        _isFetchingPrediction = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isFetchingPrediction = false;
      });
    }
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(msg),
          backgroundColor: pinkStart,
          behavior: SnackBarBehavior.floating,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
      );

  Future<void> _downloadPDF() async {
    try {
      final regularFont = await PdfGoogleFonts.notoSansRegular();
      final boldFont = await PdfGoogleFonts.notoSansBold();
      final pdf = pw.Document(
        theme: pw.ThemeData.withFont(
          base: regularFont,
          bold: boldFont,
        ),
      );

      pdf.addPage(
        pw.MultiPage(
          build: (context) => [
            pw.Text(
              'Health Report',
              style: pw.TextStyle(
                fontSize: 24,
                fontWeight: pw.FontWeight.bold,
              ),
            ),
            pw.SizedBox(height: 20),
            pw.Text('Patient Name: ${reportData['patientName']}'),
            pw.Text('Patient ID: ${reportData['patientId']}'),
            pw.Text('Age: ${reportData['age']}'),
            pw.Text('Date: ${reportData['date']}'),
            pw.SizedBox(height: 20),
            pw.Text(
              'Risk Assessment',
              style: pw.TextStyle(
                fontSize: 18,
                fontWeight: pw.FontWeight.bold,
              ),
            ),
            pw.SizedBox(height: 10),
            ...((reportData['riskAssessment'] as List).map(
              (risk) => pw.Padding(
                padding: const pw.EdgeInsets.only(bottom: 8),
                child: pw.Text(
                  '${risk['condition']} : ${(risk['probability'] * 100).toStringAsFixed(1)}%',
                ),
              ),
            )),
            pw.SizedBox(height: 20),
            pw.Text(
              'Recommendations',
              style: pw.TextStyle(
                fontSize: 18,
                fontWeight: pw.FontWeight.bold,
              ),
            ),
            pw.SizedBox(height: 10),
            ...((reportData['recommendations'] as List).map(
              (rec) => pw.Bullet(text: rec.toString()),
            )),
          ],
        ),
      );

      final bytes = await pdf.save();
      await Printing.sharePdf(
        bytes: bytes,
        filename: 'health_report.pdf',
      );

      _snack('PDF generated successfully');
    } catch (e, stackTrace) {
      debugPrint('Failed to generate PDF: $e');
      debugPrintStack(stackTrace: stackTrace);
      _snack('Failed to generate PDF: $e');
    }
  }

  List<Widget> _buildAiSummaryContent(String reportText) {
  const double baseFontSize = 14;

  final lines = reportText.split('\n');

  // Known section headings
  const headings = [
    'Summary:',
    'What the risk scores mean:',
    'Possible contributing factors from current profile:',
    'Action plan for next 2 weeks:',
    'When to seek medical review:',
    'Medical disclaimer:',
  ];

  return lines.map((line) {
    final trimmed = line.trim();

    if (trimmed.isEmpty) {
      return const SizedBox(height: 8);
    }

    // Remove markdown bold markers if present
    final cleanText = trimmed.replaceAll('**', '').trim();

    // Detect heading
    final isHeading = headings.contains(cleanText);

    if (isHeading) {
      return Padding(
        padding: const EdgeInsets.only(top: 10, bottom: 6),
        child: Text(
          cleanText,
          style: TextStyle(
            fontSize: baseFontSize + 2,
            fontWeight: FontWeight.bold,
            color: pinkStart,
            height: 1.5,
          ),
        ),
      );
    }

    // Normal paragraph/bullet text
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Text(
        cleanText,
        softWrap: true,
        style: const TextStyle(
          fontSize: baseFontSize,
          color: Colors.black87,
          height: 1.6,
        ),
      ),
    );
  }).toList();
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFFF5F8),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ΓöÇΓöÇ Overall Risk Summary Bar (retained) ΓöÇΓöÇ
                    _buildOverallRiskBar(),
                    const SizedBox(height: 20),

                    // ΓöÇΓöÇ AI Analysis card (only when report exists) ΓöÇΓöÇ
                    if (widget.reportText != null &&
                        widget.reportText!.isNotEmpty) ...[
                      _buildSectionCard(
                        'AI Health Analysis',
                        Icons.smart_toy,
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: _buildAiSummaryContent(widget.reportText!),
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],

                    _buildSectionCard('Patient Information', Icons.person,
                        _buildPatientInfo()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Risk Assessment', Icons.shield,
                        _buildRiskAssessment()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Lifestyle Factors',
                        Icons.accessibility_new, _buildLifestyle()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Health Recommendations', Icons.lightbulb,
                        _buildRecommendations()),
                    const SizedBox(height: 16),
                    _buildDisclaimer(),
                    const SizedBox(height: 24),
                    _buildActionButtons(),
                    const SizedBox(height: 16),
                    _buildBackButton(),
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

  Widget _buildHeader() {
    return Stack(
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
                    icon: const Icon(Icons.arrow_back,
                        color: Colors.white, size: 28),
                  ),
                  const SizedBox(width: 10),
                  const Icon(Icons.description, color: Colors.white, size: 40),
                ],
              ),
              const SizedBox(height: 10),
              const Text(
                'Health Report',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1,
                ),
              ),
              const Text(
                'Your comprehensive health assessment',
                style: TextStyle(color: Colors.white70, fontSize: 14),
              ),
            ],
          ),
        ),
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
    );
  }

  Widget _buildOverallRiskBar() {
    final prob = _overallProbability;
    final cfg = riskConfig(prob);
    final Color barColor = cfg['color'] as Color;
    final Color bgColor = cfg['bg'] as Color;
    final Color borderColor = cfg['border'] as Color;
    final String label = cfg['label'] as String;
    final IconData icon = cfg['icon'] as IconData;

    final segments = ['No Risk', 'Low', 'Moderate', 'High', 'Very High'];
    final segColors = [
      const Color(0xFF16A34A),
      const Color(0xFF65A30D),
      const Color(0xFFD97706),
      const Color(0xFFEA580C),
      const Color(0xFFDC2626),
    ];
    final boundaries = [0.0, 0.10, 0.30, 0.55, 0.75, 1.01];

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 2),
        boxShadow: [
          BoxShadow(
            color: pinkStart.withOpacity(0.12),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title + probability bubble
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFCE7F3),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: pinkStart, size: 28),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Overall Health Risk',
                      style: TextStyle(
                          fontSize: 12,
                          color: Colors.black45,
                          fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      label,
                      style: TextStyle(
                          fontSize: 21,
                          fontWeight: FontWeight.w800,
                          color: pinkStart),
                    ),
                  ],
                ),
              ),
              // Probability pill
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  gradient: LinearGradient(colors: [pinkStart, pinkEnd]),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                        color: pinkStart.withOpacity(0.35),
                        blurRadius: 10,
                        offset: const Offset(0, 4)),
                  ],
                ),
                child: Column(
                  children: [
                    Text(
                      prob.toStringAsFixed(2),
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.w900,
                          height: 1),
                    ),
                    const SizedBox(height: 2),
                    const Text(
                      'probability',
                      style: TextStyle(color: Colors.white70, fontSize: 10),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 20),

          // Scale labels
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              Text('0.0',
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.black38,
                      fontWeight: FontWeight.w600)),
              Text('0.25',
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.black38,
                      fontWeight: FontWeight.w600)),
              Text('0.5',
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.black38,
                      fontWeight: FontWeight.w600)),
              Text('0.75',
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.black38,
                      fontWeight: FontWeight.w600)),
              Text('1.0',
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.black38,
                      fontWeight: FontWeight.w600)),
            ],
          ),
          const SizedBox(height: 6),

          // Animated gradient bar
          LayoutBuilder(
            builder: (context, constraints) {
              final totalWidth = constraints.maxWidth;
              return AnimatedBuilder(
                animation: _barAnimation,
                builder: (context, _) {
                  final animatedProb =
                      (prob * _barAnimation.value).clamp(0.0, 1.0);
                  final thumbLeft = (totalWidth * animatedProb - 14)
                      .clamp(0.0, totalWidth - 28);
                  return SizedBox(
                    height: 32,
                    child: Stack(
                      clipBehavior: Clip.none,
                      children: [
                        // Full gradient track
                        Positioned(
                          top: 9,
                          left: 0,
                          right: 0,
                          child: Container(
                            height: 14,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(20),
                              gradient: const LinearGradient(
                                colors: [
                                  Color(0xFF16A34A),
                                  Color(0xFF65A30D),
                                  Color(0xFFD97706),
                                  Color(0xFFEA580C),
                                  Color(0xFFDC2626),
                                ],
                              ),
                            ),
                          ),
                        ),
                        // White overlay mask (unfilled portion)
                        Positioned(
                          top: 9,
                          left: totalWidth * animatedProb,
                          right: 0,
                          child: Container(
                            height: 14,
                            decoration: const BoxDecoration(
                              color: Color(0xFFFFF5F8),
                              borderRadius: BorderRadius.only(
                                topRight: Radius.circular(20),
                                bottomRight: Radius.circular(20),
                              ),
                            ),
                          ),
                        ),
                        // Thumb ΓÇö pink
                        Positioned(
                          left: thumbLeft,
                          top: 0,
                          child: Container(
                            width: 32,
                            height: 32,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                              border: Border.all(color: pinkStart, width: 3),
                              boxShadow: [
                                BoxShadow(
                                    color: pinkStart.withOpacity(0.40),
                                    blurRadius: 10,
                                    offset: const Offset(0, 3)),
                              ],
                            ),
                            child: Center(
                              child: Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                    color: pinkStart, shape: BoxShape.circle),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              );
            },
          ),

          const SizedBox(height: 14),

          // 5 category chips
          Row(
            children: List.generate(5, (i) {
              final isActive =
                  prob >= boundaries[i] && prob < boundaries[i + 1];
              return Expanded(
                child: Container(
                  margin: EdgeInsets.only(right: i < 4 ? 4 : 0),
                  padding: const EdgeInsets.symmetric(vertical: 7),
                  decoration: BoxDecoration(
                    gradient: isActive
                        ? LinearGradient(colors: [pinkStart, pinkEnd])
                        : null,
                    color: isActive ? null : const Color(0xFFFCE7F3),
                    borderRadius: BorderRadius.circular(8),
                    border: isActive
                        ? null
                        : Border.all(color: const Color(0xFFFCE7F3)),
                  ),
                  child: Text(
                    segments[i],
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 9,
                      fontWeight: isActive ? FontWeight.w800 : FontWeight.w500,
                      color:
                          isActive ? Colors.white : pinkStart.withOpacity(0.6),
                    ),
                  ),
                ),
              );
            }),
          ),

          const SizedBox(height: 16),
          const Divider(color: Color(0xFFFCE7F3), thickness: 1.5),
          const SizedBox(height: 12),

          // Per-condition mini pills
          const Text(
            'Conditions assessed:',
            style: TextStyle(
                fontSize: 12,
                color: Colors.black54,
                fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: (reportData['riskAssessment'] as List).map((r) {
              final p = (r['probability'] as num).toDouble();
              final c = riskConfig(p);
              final col = c['color'] as Color;
              final lbl = c['label'] as String;
              return Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: const Color(0xFFFCE7F3),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFFFBCFE8)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                        width: 8,
                        height: 8,
                        decoration:
                            BoxDecoration(color: col, shape: BoxShape.circle)),
                    const SizedBox(width: 6),
                    Text(r['condition'] as String,
                        style: const TextStyle(
                            fontSize: 11,
                            color: Colors.black87,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(width: 4),
                    Text('ΓÇó',
                        style: TextStyle(
                            fontSize: 11, color: pinkEnd.withOpacity(0.7))),
                    const SizedBox(width: 4),
                    Text('${p.toStringAsFixed(2)} ┬╖ $lbl',
                        style: TextStyle(
                            fontSize: 11,
                            color: col,
                            fontWeight: FontWeight.w700)),
                  ],
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  // ΓöÇΓöÇ Patient Info ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildPatientInfo() {
    return Column(
      children: [
        _buildInfoItem('Name', reportData['patientName']),
        _buildInfoItem('Patient ID', reportData['patientId']),
        _buildInfoItem('Age', reportData['age']),
        _buildInfoItem('Report Date', reportData['date']),
      ],
    );
  }

  Widget _buildInfoItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 2,
            child: Text(label,
                style: const TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                    fontWeight: FontWeight.w500)),
          ),
          const SizedBox(width: 12),
          Expanded(
            flex: 3,
            child: Text(
              value,
              textAlign: TextAlign.right,
              softWrap: true,
              style: const TextStyle(
                  fontSize: 14,
                  color: Colors.black87,
                  fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }

  // ΓöÇΓöÇ Symptoms ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildSymptoms() {
    return Column(
      children: (reportData['symptoms'] as List).map((s) {
        return _buildSymptomItem(s['name'], s['value']);
      }).toList(),
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
          Text(name,
              style: const TextStyle(fontSize: 14, color: Colors.black87)),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: pinkStart.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: pinkStart.withOpacity(0.3)),
            ),
            child: Text(value,
                style: TextStyle(
                    fontSize: 13,
                    color: pinkStart,
                    fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  // ΓöÇΓöÇ Risk Assessment cards ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildRiskAssessment() {
    final risks = reportData['riskAssessment'] as List;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildRiskLegend(),
        const SizedBox(height: 16),
        ...risks.map((r) => _buildRiskCard(
              r['condition'] as String,
              (r['probability'] as num).toDouble(),
            )),
      ],
    );
  }

  Widget _buildRiskLegend() {
    final levels = [
      {'label': 'No Risk', 'color': const Color(0xFF16A34A)},
      {'label': 'Low', 'color': const Color(0xFF15803D)},
      {'label': 'Moderate', 'color': const Color(0xFFB45309)},
      {'label': 'High', 'color': const Color(0xFFEA580C)},
      {'label': 'Very High', 'color': const Color(0xFFDC2626)},
    ];
    return Wrap(
      spacing: 10,
      runSpacing: 6,
      children: levels.map((l) {
        final c = l['color'] as Color;
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(color: c, shape: BoxShape.circle)),
            const SizedBox(width: 4),
            Text(l['label'] as String,
                style: TextStyle(
                    fontSize: 11, color: c, fontWeight: FontWeight.w600)),
          ],
        );
      }).toList(),
    );
  }

  Widget _buildRiskCard(String condition, double probability) {
    final cfg = riskConfig(probability);
    final color = cfg['color'] as Color;
    final bg = cfg['bg'] as Color;
    final border = cfg['border'] as Color;
    final label = cfg['label'] as String;
    final icon = cfg['icon'] as IconData;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
        boxShadow: [
          BoxShadow(
              color: pinkStart.withOpacity(0.08),
              blurRadius: 8,
              offset: const Offset(0, 3)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              // Icon ΓÇö semantic color in tinted box
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 18),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(condition,
                    style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                    color: Colors.black87),
                  softWrap: true),
              ),
              // Risk label badge
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border:
                      Border.all(color: color.withOpacity(0.40), width: 1.5),
                ),
                child: Text(label,
                    style: TextStyle(
                        fontSize: 11,
                        color: color,
                        fontWeight: FontWeight.bold)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Align(
            alignment: Alignment.centerRight,
            child: Text(
              'Probability: ${probability.toStringAsFixed(2)}',
              style: TextStyle(
                  fontSize: 12, color: color, fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }

  // ΓöÇΓöÇ Lifestyle ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildLifestyle() {
    final ls = reportData['lifestyle'] as Map<String, dynamic>;
    return Column(
      children: [
        _buildLifestyleItem(Icons.fitness_center, 'Exercise', ls['exercise']),
        _buildLifestyleItem(Icons.bedtime, 'Sleep', ls['sleep']),
        _buildLifestyleItem(Icons.water_drop, 'Water Intake', ls['water']),
        _buildLifestyleItem(Icons.psychology, 'Stress Level', ls['stress']),
      ],
    );
  }

  Widget _buildLifestyleItem(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
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
            child: Text(label,
                style: const TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                    fontWeight: FontWeight.w500)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              softWrap: true,
              style: const TextStyle(
                  fontSize: 14,
                  color: Colors.black87,
                  fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }

  // ΓöÇΓöÇ Recommendations ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildRecommendations() {
    return Column(
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
                decoration:
                    BoxDecoration(color: pinkStart, shape: BoxShape.circle),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(rec,
                    style: const TextStyle(
                        fontSize: 14, color: Colors.black87, height: 1.5)),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  // ΓöÇΓöÇ Disclaimer ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFCE7F3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFBCFE8), width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.info_outline, color: pinkStart, size: 20),
              const SizedBox(width: 8),
              const Text('Disclaimer',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'This report provides risk predictions and preventive advice based on '
            'self-reported data. It is not a medical diagnosis. Please consult '
            'with a healthcare professional for proper medical evaluation.',
            style: TextStyle(fontSize: 12, color: Colors.black87, height: 1.5),
          ),
        ],
      ),
    );
  }

  // ✅ Only Download PDF button
Widget _buildActionButtons() {
  return SizedBox(
    width: double.infinity,
    child: _buildActionButton(
      icon: Icons.picture_as_pdf,
      label: 'Download PDF',
      onTap: _downloadPDF,
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
              end: Alignment.bottomRight),
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
                color: pinkStart.withOpacity(0.3),
                blurRadius: 8,
                offset: const Offset(0, 4))
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 28),
            const SizedBox(height: 6),
            Text(label,
                style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 12),
                textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }

  // ΓöÇΓöÇ Back Button ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Widget _buildBackButton() {
    return OutlinedButton.icon(
      onPressed: () => Navigator.pop(context),
      icon: Icon(Icons.home, color: pinkStart),
      label: Text('Back to Dashboard',
          style: TextStyle(
              color: pinkStart, fontWeight: FontWeight.bold, fontSize: 16)),
      style: OutlinedButton.styleFrom(
        side: BorderSide(color: pinkStart, width: 2),
        minimumSize: const Size(double.infinity, 54),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  // ΓöÇΓöÇ Section Card wrapper ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
              offset: const Offset(0, 4))
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
              Text(title,
                  style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87)),
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
}
