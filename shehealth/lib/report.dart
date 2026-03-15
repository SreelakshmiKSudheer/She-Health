import 'package:flutter/material.dart';

import 'models/app_models.dart';

class HealthReportPage extends StatelessWidget {
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

  List<Map<String, dynamic>> get _riskItems {
    final raw = predictionData?['predictions'];
    if (raw is! Map<String, dynamic>) {
      return [];
    }

    return raw.entries.map((entry) {
      final value = entry.value as Map<String, dynamic>;
      final probPct = (value['probability'] as num? ?? 0).toDouble();
      return {
        'condition': entry.key,
        'probability': (probPct / 100).clamp(0.0, 1.0),
        'label': value['label'] as String? ?? 'Unknown',
      };
    }).toList()
      ..sort((a, b) =>
          (b['probability'] as double).compareTo(a['probability'] as double));
  }

  double get _overallProbability {
    if (_riskItems.isEmpty) {
      return 0;
    }
    return _riskItems
        .map((e) => e['probability'] as double)
        .reduce((a, b) => a > b ? a : b);
  }

  String get _riskCategory {
    final p = _overallProbability;
    if (p < 0.10) return 'No Risk';
    if (p < 0.30) return 'Low Risk';
    if (p < 0.55) return 'Moderate Risk';
    if (p < 0.75) return 'High Risk';
    return 'Very High Risk';
  }

  Color get _riskColor {
    final p = _overallProbability;
    if (p < 0.10) return const Color(0xFF16A34A);
    if (p < 0.30) return const Color(0xFFC85A7A);
    if (p < 0.55) return const Color(0xFFB8436A);
    if (p < 0.75) return const Color(0xFF9E2D57);
    return const Color(0xFF7B1D3F);
  }

  List<String> get _recommendations {
    final p = _overallProbability;
    if (p < 0.30) {
      return const [
        'Continue regular cycle tracking and maintain healthy routines.',
        'Keep hydration and sleep quality consistent.',
        'Follow periodic preventive checkups.',
      ];
    }

    if (p < 0.55) {
      return const [
        'Monitor symptoms weekly and note any worsening patterns.',
        'Adopt stress-reduction habits and balanced nutrition.',
        'Discuss these results with a clinician during your next visit.',
      ];
    }

    return const [
      'Schedule a medical consultation for clinical evaluation.',
      'Track symptom severity daily and share trends with your doctor.',
      'Follow a physician-guided plan for diagnostics and lifestyle changes.',
    ];
  }

  @override
  Widget build(BuildContext context) {
    final hasPrediction = _riskItems.isNotEmpty;

    return Scaffold(
      backgroundColor: const Color(0xFFFFF5F8),
      appBar: AppBar(
        backgroundColor: const Color(0xFFC85A7A),
        foregroundColor: Colors.white,
        title: const Text('Health Report'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildSummaryCard(),
            const SizedBox(height: 14),
            _buildPatientCard(),
            const SizedBox(height: 14),
            _buildSectionCard(
              title: 'Condition-wise Risk Assessment',
              child: hasPrediction
                  ? Column(
                      children: _riskItems
                          .map(
                            (r) => _buildRiskRow(
                              condition: r['condition'] as String,
                              probability: r['probability'] as double,
                              label: r['label'] as String,
                            ),
                          )
                          .toList(),
                    )
                  : const Text('No prediction data available yet.'),
            ),
            const SizedBox(height: 14),
            _buildSectionCard(
              title: 'Recommendations',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: _recommendations
                    .map(
                      (r) => Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Padding(
                              padding: EdgeInsets.only(top: 6),
                              child: Icon(Icons.circle,
                                  size: 7, color: Color(0xFFC85A7A)),
                            ),
                            const SizedBox(width: 8),
                            Expanded(child: Text(r)),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
            if (reportText != null && reportText!.trim().isNotEmpty) ...[
              const SizedBox(height: 14),
              _buildSectionCard(
                title: 'AI Narrative',
                child: Text(
                  reportText!,
                  style: const TextStyle(height: 1.5),
                ),
              ),
            ],
            const SizedBox(height: 14),
            _buildSectionCard(
              title: 'Medical Disclaimer',
              child: const Text(
                'This report is a predictive screening aid based on questionnaire data. '
                'It is not a diagnosis. Please consult a qualified healthcare professional '
                'for definitive medical advice and treatment decisions.',
                style: TextStyle(height: 1.5),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Overall Risk Summary',
            style: TextStyle(
                color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            _riskCategory,
            style: const TextStyle(
                color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(
            'Overall probability: ${_overallProbability.toStringAsFixed(2)}',
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: _overallProbability,
              minHeight: 9,
              color: Colors.white,
              backgroundColor: Colors.white.withOpacity(0.3),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPatientCard() {
    return _buildSectionCard(
      title: 'Patient Information',
      child: Column(
        children: [
          _infoRow('Name', localUser?.fullName ?? 'Unknown'),
          _infoRow('User ID', userId ?? localUser?.userId ?? 'Not available'),
          _infoRow('Email', localUser?.email ?? 'Not available'),
          _infoRow(
            'Height / Weight',
            (localUser?.heightCm != null && localUser?.weightKg != null)
                ? '${localUser!.heightCm!.toStringAsFixed(1)} cm / ${localUser!.weightKg!.toStringAsFixed(1)} kg'
                : 'Not available',
          ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: const TextStyle(
                color: Colors.grey,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskRow({
    required String condition,
    required double probability,
    required String label,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _riskColor.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _riskColor.withOpacity(0.18)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  condition,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                  ),
                ),
              ),
              Text(
                label,
                style: TextStyle(
                  color: _riskColor,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: probability,
            minHeight: 8,
            color: _riskColor,
            backgroundColor: const Color(0xFFFCE7F3),
          ),
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerRight,
            child: Text(
              probability.toStringAsFixed(2),
              style: const TextStyle(fontSize: 12, color: Colors.black54),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionCard({required String title, required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}
