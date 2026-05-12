import 'package:flutter/material.dart';

import 'models/app_models.dart';
import 'services/backend_api_service.dart';
import 'services/local_storage_service.dart';

class ThyroidReportPage extends StatefulWidget {
  final String? userId;
  final Map<String, dynamic>? predictionData;
  final LocalUserProfile? localUser;
  final String? reportText;

  const ThyroidReportPage({
    super.key,
    this.userId,
    this.predictionData,
    this.localUser,
    this.reportText,
  });

  @override
  State<ThyroidReportPage> createState() => _ThyroidReportPageState();
}

class _ThyroidReportPageState extends State<ThyroidReportPage> {
  final BackendApiService _api = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;

  Map<String, dynamic>? _predictionData;
  LocalUserProfile? _localUser;
  bool _isLoading = true;
  String? _loadError;

  @override
  void initState() {
    super.initState();
    _predictionData = widget.predictionData;
    _localUser = widget.localUser;
    _loadReportData();
  }

  Future<void> _loadReportData() async {
    if (_predictionData != null) {
      if (_localUser == null && widget.userId != null && widget.userId!.isNotEmpty) {
        _localUser = await _localStorage.findByUserId(widget.userId!);
      }
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoading = false;
      });
      return;
    }

    final userId = widget.userId;
    if (userId == null || userId.isEmpty) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoading = false;
        _loadError = 'No user id available for thyroid report.';
      });
      return;
    }

    try {
      final latest = await _api.getLatestThyroidPrediction(userId);
      final localUser = await _localStorage.findByUserId(userId);
      if (!mounted) {
        return;
      }
      setState(() {
        _predictionData = latest;
        _localUser = localUser;
        _isLoading = false;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoading = false;
        _loadError = error.toString();
      });
    }
  }

  static const Color kPink = Color(0xFFC85A7A);
  static const Color kPinkSoft = Color(0xFFE59393);
  static const Color kPinkLight = Color(0xFFFFF5F8);

  Map<String, dynamic>? get _thyroidResult {
    final raw = _predictionData?['predictions'];
    if (raw is! Map<String, dynamic>) {
      return null;
    }

    final direct = raw['Thyroid'];
    if (direct is Map<String, dynamic>) {
      return direct;
    }

    if (raw.isEmpty) {
      return null;
    }

    final first = raw.entries.first.value;
    return first is Map<String, dynamic> ? first : null;
  }

  double _probabilityPct() {
    final result = _thyroidResult;
    final value = result?['probability'];
    if (value is num) {
      return value.toDouble();
    }
    return 0.0;
  }

  int _riskLevel() {
    final result = _thyroidResult;
    final value = result?['risk_level'];
    if (value is num) {
      return value.toInt();
    }
    return 1;
  }

  String _categoryLabel() {
    final result = _thyroidResult;
    return result?['category_name']?.toString() ?? 'Unknown';
  }

  String _statusLabel() {
    final result = _thyroidResult;
    return result?['status']?.toString() ?? 'unknown';
  }

  Map<String, dynamic> _riskConfig(double p) {
    if (p < 10) {
      return {'label': 'Very Low Risk', 'color': const Color(0xFF9D8EC7), 'bg': const Color(0xFFF3F0FB), 'border': const Color(0xFFD4CCF0), 'icon': Icons.check_circle_outline};
    }
    if (p < 25) {
      return {'label': 'Low Risk', 'color': kPink, 'bg': const Color(0xFFFFF5F8), 'border': const Color(0xFFFCE7F3), 'icon': Icons.thumb_up_outlined};
    }
    if (p < 50) {
      return {'label': 'Moderate Risk', 'color': const Color(0xFFB8436A), 'bg': const Color(0xFFFEECF3), 'border': const Color(0xFFF9B8D0), 'icon': Icons.warning_amber_outlined};
    }
    if (p < 75) {
      return {'label': 'High Risk', 'color': const Color(0xFF9E2D57), 'bg': const Color(0xFFFCE4EE), 'border': const Color(0xFFF4A0BF), 'icon': Icons.error_outline};
    }
    return {'label': 'Very High Risk', 'color': const Color(0xFF7B1D3F), 'bg': const Color(0xFFF9D6E4), 'border': const Color(0xFFEC87AD), 'icon': Icons.dangerous_outlined};
  }

  List<String> _recommendations() {
    final risk = _probabilityPct();
    if (risk < 10) {
      return [
        'Keep tracking symptoms and review if anything changes.',
        'Maintain balanced meals, hydration, and regular sleep.',
        'Repeat thyroid screening on the schedule advised by your clinician.',
      ];
    }
    if (risk < 50) {
      return [
        'Book a thyroid review if fatigue, weight change, or temperature sensitivity persist.',
        'Keep a daily log of energy, sleep, and medication use.',
        'Discuss lab testing or medication changes with a healthcare professional.',
      ];
    }
    return [
      'Seek clinical review soon if symptoms are worsening or new symptoms appear.',
      'Bring your thyroid assessment history to your next appointment.',
      'Do not change medication without medical guidance.',
    ];
  }

  String _summaryText() {
    final risk = _probabilityPct();
    final label = _categoryLabel();
    if (risk == 0 && _thyroidResult == null) {
      return widget.reportText ?? 'No thyroid prediction is available yet. Complete the assessment to generate a thyroid report.';
    }
    return widget.reportText ??
        'Your thyroid assessment returned a $label result with a probability of ${risk.toStringAsFixed(2)}%. This is a screening result, not a diagnosis.';
  }

  Widget _sectionTitle(String title, {String? subtitle}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(color: Color(0xFF4A2032), fontSize: 18, fontWeight: FontWeight.w800),
          ),
          if (subtitle != null) ...[
            const SizedBox(height: 4),
            Text(subtitle, style: const TextStyle(color: Color(0xFF7A5C6A), fontSize: 13, height: 1.4)),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        backgroundColor: kPinkLight,
        appBar: AppBar(
          backgroundColor: kPink,
          foregroundColor: Colors.white,
          title: const Text('Thyroid Report'),
          elevation: 0,
        ),
        body: const Center(child: CircularProgressIndicator(color: kPink)),
      );
    }

    if (_loadError != null && _predictionData == null) {
      return Scaffold(
        backgroundColor: kPinkLight,
        appBar: AppBar(
          backgroundColor: kPink,
          foregroundColor: Colors.white,
          title: const Text('Thyroid Report'),
          elevation: 0,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(
              _loadError!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Color(0xFF4A2032), fontSize: 14),
            ),
          ),
        ),
      );
    }

    final probability = _probabilityPct();
    final riskConfig = _riskConfig(probability);
    final comparison = _predictionData?['comparison'];

    return Scaffold(
      backgroundColor: kPinkLight,
      appBar: AppBar(
        backgroundColor: kPink,
        foregroundColor: Colors.white,
        title: const Text('Thyroid Report'),
        elevation: 0,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
          children: [
            Container(
              padding: const EdgeInsets.all(22),
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [kPink, kPinkSoft], begin: Alignment.topLeft, end: Alignment.bottomRight),
                borderRadius: BorderRadius.circular(28),
                boxShadow: [
                  BoxShadow(color: kPink.withOpacity(0.18), blurRadius: 24, offset: const Offset(0, 14)),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Thyroid Assessment',
                    style: TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.w800),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'A dedicated report for the separate thyroid assessment pipeline.',
                    style: TextStyle(color: Colors.white.withOpacity(0.92), fontSize: 14, height: 1.45),
                  ),
                  const SizedBox(height: 18),
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(color: Colors.white.withOpacity(0.16)),
                    ),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(color: Colors.white.withOpacity(0.14), shape: BoxShape.circle),
                          child: Icon(riskConfig['icon'] as IconData, color: Colors.white),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Current Risk', style: TextStyle(color: Colors.white70, fontSize: 12)),
                              const SizedBox(height: 4),
                              Text(
                                '${riskConfig['label']}',
                                style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800),
                              ),
                            ],
                          ),
                        ),
                        Text(
                          '${probability.toStringAsFixed(2)}%',
                          style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.w900),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            if ((_statusLabel()).isNotEmpty && _statusLabel() != 'ok')
              Container(
                margin: const EdgeInsets.only(bottom: 16),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF7E6),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFF1D59B)),
                ),
                child: Text(
                  'Prediction status: ${_statusLabel()}',
                  style: const TextStyle(color: Color(0xFF8A5A12), fontWeight: FontWeight.w600),
                ),
              ),
            if (comparison is Map<String, dynamic>) ...[
              _sectionTitle('Trend', subtitle: 'How this assessment compares with the previous thyroid prediction.'),
              ...[
                Builder(
                  builder: (context) {
                    final changed = comparison['changed_diseases'];
                    String details = 'No previous comparison details available.';
                    if (changed is List && changed.isNotEmpty) {
                      details = changed.map((entry) {
                        if (entry is Map<String, dynamic>) {
                          final disease = entry['disease']?.toString() ?? 'Thyroid';
                          final previousLevel = entry['previous_level']?.toString() ?? '0';
                          final currentLevel = entry['current_level']?.toString() ?? '0';
                          return '$disease: $previousLevel → $currentLevel';
                        }
                        return entry.toString();
                      }).join('\n');
                    }

                    return Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: const Color(0xFFF3DAE2)),
                      ),
                      child: Text(
                        'Change type: ${comparison['change_type'] ?? 'unknown'}\n$details',
                        style: const TextStyle(color: Color(0xFF6C5060), height: 1.5),
                      ),
                    );
                  },
                ),
              ],
              const SizedBox(height: 16),
            ],
            _sectionTitle('Summary', subtitle: 'A concise interpretation of the thyroid screening result.'),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF3DAE2)),
              ),
              child: Text(
                _summaryText(),
                style: const TextStyle(color: Color(0xFF4A2032), height: 1.55),
              ),
            ),
            const SizedBox(height: 16),
            _sectionTitle('Assessment Details'),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF3DAE2)),
              ),
              child: Column(
                children: [
                  _detailRow('Disease', 'Thyroid'),
                  _detailRow('Probability', '${probability.toStringAsFixed(2)}%'),
                  _detailRow('Risk level', _riskLevel().toString()),
                  _detailRow('Category', _categoryLabel()),
                  _detailRow('Status', _statusLabel()),
                  _detailRow('User ID', widget.userId ?? 'Not available'),
                ],
              ),
            ),
            const SizedBox(height: 16),
            _sectionTitle('Recommendations', subtitle: 'Practical next steps based on the current result.'),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF3DAE2)),
              ),
              child: Column(
                children: _recommendations()
                    .map(
                      (item) => Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('• ', style: TextStyle(color: Color(0xFFC85A7A), fontWeight: FontWeight.w800)),
                            Expanded(
                              child: Text(
                                item,
                                style: const TextStyle(color: Color(0xFF4A2032), height: 1.45),
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
            const SizedBox(height: 16),
            _sectionTitle('Profile Snapshot'),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF3DAE2)),
              ),
              child: Column(
                children: [
                  _detailRow('Name', _localUser?.fullName ?? 'Unknown'),
                  _detailRow('Date of birth', _localUser?.dob ?? 'Not provided'),
                  _detailRow('Marital status', _localUser?.maritalStatus ?? 'Not provided'),
                  _detailRow('Activity level', _localUser?.activityLevel ?? 'Not provided'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: const TextStyle(color: Color(0xFF7A5C6A), fontWeight: FontWeight.w600),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(color: Color(0xFF4A2032), fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }
}
