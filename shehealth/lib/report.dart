import 'package:flutter/material.dart';

class HealthReportPage extends StatefulWidget {
  final String? reportText;
  const HealthReportPage({super.key, this.reportText});

  @override
  State<HealthReportPage> createState() => _HealthReportPageState();
}

class _HealthReportPageState extends State<HealthReportPage>
    with SingleTickerProviderStateMixin {
  final Color pinkStart = const Color(0xFFC85A7A);
  final Color pinkEnd   = const Color(0xFFE59393);

  late AnimationController _animController;
  late Animation<double>   _barAnimation;

  // Risk config — all pink palette, intensity shows severity
  static Map<String, dynamic> riskConfig(double p) {
    if (p < 0.10) {
      return {
        'label': 'No Risk',
        'color': const Color(0xFF9D8EC7),   // soft lavender
        'bg':    const Color(0xFFF3F0FB),
        'border':const Color(0xFFD4CCF0),
        'icon':  Icons.check_circle_outline,
        'intensity': 0,
      };
    } else if (p < 0.30) {
      return {
        'label': 'Low Risk',
        'color': const Color(0xFFC85A7A),   // app pink primary
        'bg':    const Color(0xFFFFF5F8),
        'border':const Color(0xFFFCE7F3),
        'icon':  Icons.thumb_up_outlined,
        'intensity': 1,
      };
    } else if (p < 0.55) {
      return {
        'label': 'Moderate Risk',
        'color': const Color(0xFFB8436A),   // medium deep pink
        'bg':    const Color(0xFFFEECF3),
        'border':const Color(0xFFF9B8D0),
        'icon':  Icons.warning_amber_outlined,
        'intensity': 2,
      };
    } else if (p < 0.75) {
      return {
        'label': 'High Risk',
        'color': const Color(0xFF9E2D57),   // deep rose
        'bg':    const Color(0xFFFCE4EE),
        'border':const Color(0xFFF4A0BF),
        'icon':  Icons.error_outline,
        'intensity': 3,
      };
    } else {
      return {
        'label': 'Very High Risk',
        'color': const Color(0xFF7B1D3F),   // dark crimson rose
        'bg':    const Color(0xFFF9D6E4),
        'border':const Color(0xFFEC87AD),
        'icon':  Icons.dangerous_outlined,
        'intensity': 4,
      };
    }
  }
  // ── Sample data ──────────────────────────────
  final Map<String, dynamic> reportData = {
    'patientName': 'Sarah Anderson',
    'patientId':   'SH2024001',
    'date':        'October 12, 2025',
    'age':         '28 years',
    'assessmentDate': 'October 10, 2025',

    'symptoms': [
      {'name': 'Cycle Regularity',   'value': 'Regular'},
      {'name': 'Period Pain Level',  'value': 'Mild'},
      {'name': 'Weight Changes',     'value': 'None'},
      {'name': 'Fatigue Frequency',  'value': 'Sometimes'},
      {'name': 'Mood Swing Pattern', 'value': 'Occasional'},
    ],

    'riskAssessment': [
      {'condition': 'PCOD / PCOS',       'probability': 0.15},
      {'condition': 'Thyroid Disorders', 'probability': 0.05},
      {'condition': 'Endometriosis',     'probability': 0.42},
      {'condition': 'Cervical Cancer',   'probability': 0.03},
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
      'sleep':    '7-8 hours',
      'water':    '1.5-2L daily',
      'stress':   'Moderate',
    },
  };

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
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
      );

  void _downloadPDF() => _snack('PDF download functionality will be added');
  void _shareReport()  => _snack('Share report functionality will be added');
  void _emailReport()  => _snack('Email report functionality will be added');
  void _printReport()  => _snack('Print report functionality will be added');

  // ── BUILD ────────────────────────────────────
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
                    // ── Overall Risk Summary Bar (retained) ──
                    _buildOverallRiskBar(),
                    const SizedBox(height: 20),

                    // ── AI Analysis card (only when report exists) ──
                    if (widget.reportText != null && widget.reportText!.isNotEmpty) ...[
                      _buildSectionCard(
                        'AI Health Analysis',
                        Icons.smart_toy,
                        Text(
                          widget.reportText!,
                          style: const TextStyle(
                              fontSize: 14, color: Colors.black87, height: 1.6),
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],

                    _buildSectionCard('Patient Information', Icons.person,
                        _buildPatientInfo()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Symptom Summary', Icons.favorite,
                        _buildSymptoms()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Risk Assessment', Icons.shield,
                        _buildRiskAssessment()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Lifestyle Factors',
                        Icons.accessibility_new, _buildLifestyle()),
                    const SizedBox(height: 16),
                    _buildSectionCard('Health Recommendations',
                        Icons.lightbulb, _buildRecommendations()),
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

  // ── Header ───────────────────────────────────
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

  // ══════════════════════════════════════════════
  //  OVERALL RISK SUMMARY BAR  (retained as-is)
  // ══════════════════════════════════════════════
  Widget _buildOverallRiskBar() {
    final prob        = _overallProbability;
    final cfg         = riskConfig(prob);
    final Color barColor   = cfg['color']  as Color;
    final Color bgColor    = cfg['bg']     as Color;
    final Color borderColor= cfg['border'] as Color;
    final String label     = cfg['label']  as String;
    final IconData icon    = cfg['icon']   as IconData;

    final segments  = ['No Risk', 'Low', 'Moderate', 'High', 'Very High'];
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
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
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
              Text('0.0',  style: TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w600)),
              Text('0.25', style: TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w600)),
              Text('0.5',  style: TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w600)),
              Text('0.75', style: TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w600)),
              Text('1.0',  style: TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w600)),
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
                  final thumbLeft =
                      (totalWidth * animatedProb - 14).clamp(0.0, totalWidth - 28);
                  return SizedBox(
                    height: 32,
                    child: Stack(
                      clipBehavior: Clip.none,
                      children: [
                        // Full gradient track
                        Positioned(
                          top: 9, left: 0, right: 0,
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
                        // Thumb — pink
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
              final isActive = prob >= boundaries[i] && prob < boundaries[i + 1];
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
                      color: isActive ? Colors.white : pinkStart.withOpacity(0.6),
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
                fontSize: 12, color: Colors.black54, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: (reportData['riskAssessment'] as List).map((r) {
              final p   = (r['probability'] as num).toDouble();
              final c   = riskConfig(p);
              final col = c['color'] as Color;
              final lbl = c['label'] as String;
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: const Color(0xFFFCE7F3),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFFFBCFE8)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                        width: 8, height: 8,
                        decoration: BoxDecoration(color: col, shape: BoxShape.circle)),
                    const SizedBox(width: 6),
                    Text(r['condition'] as String,
                        style: const TextStyle(
                            fontSize: 11,
                            color: Colors.black87,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(width: 4),
                    Text('•',
                        style: TextStyle(fontSize: 11, color: pinkEnd.withOpacity(0.7))),
                    const SizedBox(width: 4),
                    Text('${p.toStringAsFixed(2)} · $lbl',
                        style: TextStyle(
                            fontSize: 11, color: col, fontWeight: FontWeight.w700)),
                  ],
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  // ── Patient Info ─────────────────────────────
  Widget _buildPatientInfo() {
    return Column(
      children: [
        _buildInfoItem('Name',        reportData['patientName']),
        _buildInfoItem('Patient ID',  reportData['patientId']),
        _buildInfoItem('Age',         reportData['age']),
        _buildInfoItem('Report Date', reportData['date']),
      ],
    );
  }

  Widget _buildInfoItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 14, color: Colors.grey, fontWeight: FontWeight.w500)),
          Text(value,
              style: const TextStyle(
                  fontSize: 14, color: Colors.black87, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  // ── Symptoms ─────────────────────────────────
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
                    fontSize: 13, color: pinkStart, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  // ── Risk Assessment cards ────────────────────
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
      {'label': 'No Risk',   'color': const Color(0xFF16A34A)},
      {'label': 'Low',       'color': const Color(0xFF15803D)},
      {'label': 'Moderate',  'color': const Color(0xFFB45309)},
      {'label': 'High',      'color': const Color(0xFFEA580C)},
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
                width: 10, height: 10,
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
    final cfg    = riskConfig(probability);
    final color  = cfg['color']  as Color;
    final bg     = cfg['bg']     as Color;
    final border = cfg['border'] as Color;
    final label  = cfg['label']  as String;
    final icon   = cfg['icon']   as IconData;

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
              // Icon — semantic color in tinted box
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
                        color: Colors.black87)),
              ),
              // Risk label badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: color.withOpacity(0.40), width: 1.5),
                ),
                child: Text(label,
                    style: TextStyle(
                        fontSize: 11, color: color, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              const Text('0.0',
                  style: TextStyle(fontSize: 10, color: Colors.grey)),
              const SizedBox(width: 6),
              Expanded(
                child: Stack(
                  children: [
                    // Track
                    Container(
                      height: 10,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: border),
                      ),
                    ),
                    // Filled — semantic gradient
                    FractionallySizedBox(
                      widthFactor: probability.clamp(0.0, 1.0),
                      child: Container(
                        height: 10,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [
                              Color(0xFF4ADE80), // light green
                              Color(0xFF16A34A), // dark green
                              Color(0xFFEAB308), // yellow
                              Color(0xFFF97316), // orange
                              Color(0xFFDC2626), // red
                            ],
                            stops: [0.0, 0.25, 0.5, 0.75, 1.0],
                          ),
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ),
                    // Thumb dot — semantic color
                    FractionallySizedBox(
                      widthFactor: probability.clamp(0.0, 1.0),
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: Container(
                          width: 14,
                          height: 14,
                          decoration: BoxDecoration(
                            color: color,
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 2),
                            boxShadow: [
                              BoxShadow(
                                  color: color.withOpacity(0.4),
                                  blurRadius: 4)
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              const Text('1.0',
                  style: TextStyle(fontSize: 10, color: Colors.grey)),
            ],
          ),
          const SizedBox(height: 6),
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

  // ── Lifestyle ────────────────────────────────
  Widget _buildLifestyle() {
    final ls = reportData['lifestyle'] as Map<String, dynamic>;
    return Column(
      children: [
        _buildLifestyleItem(Icons.fitness_center, 'Exercise',     ls['exercise']),
        _buildLifestyleItem(Icons.bedtime,         'Sleep',        ls['sleep']),
        _buildLifestyleItem(Icons.water_drop,      'Water Intake', ls['water']),
        _buildLifestyleItem(Icons.psychology,      'Stress Level', ls['stress']),
      ],
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
            child: Text(label,
                style: const TextStyle(
                    fontSize: 14, color: Colors.grey, fontWeight: FontWeight.w500)),
          ),
          Text(value,
              style: const TextStyle(
                  fontSize: 14, color: Colors.black87, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  // ── Recommendations ──────────────────────────
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

  // ── Disclaimer ───────────────────────────────
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

  // ── Action Buttons ───────────────────────────
  Widget _buildActionButtons() {
    return GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 2.5,
      children: [
        _buildActionButton(
            icon: Icons.picture_as_pdf, label: 'Download PDF', onTap: _downloadPDF),
        _buildActionButton(
            icon: Icons.share,          label: 'Share Report', onTap: _shareReport),
        _buildActionButton(
            icon: Icons.email,          label: 'Email Report', onTap: _emailReport),
        _buildActionButton(
            icon: Icons.print,          label: 'Print Report', onTap: _printReport),
      ],
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

  // ── Back Button ──────────────────────────────
  Widget _buildBackButton() {
    return OutlinedButton.icon(
      onPressed: () => Navigator.pop(context),
      icon: Icon(Icons.home, color: pinkStart),
      label: Text('Back to Dashboard',
          style: TextStyle(
              color: pinkStart,
              fontWeight: FontWeight.bold,
              fontSize: 16)),
      style: OutlinedButton.styleFrom(
        side: BorderSide(color: pinkStart, width: 2),
        minimumSize: const Size(double.infinity, 54),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  // ── Section Card wrapper ─────────────────────
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