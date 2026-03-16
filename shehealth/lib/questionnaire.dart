import 'package:flutter/material.dart';

import 'models/app_models.dart';
import 'report.dart';
import 'services/backend_api_service.dart';
import 'services/groq_service.dart';
import 'services/local_storage_service.dart';

class SymptomQuestionnaire extends StatefulWidget {
  final String userId;

  const SymptomQuestionnaire({super.key, required this.userId});

  @override
  State<SymptomQuestionnaire> createState() => _SymptomQuestionnaireState();
}

class _SymptomQuestionnaireState extends State<SymptomQuestionnaire> {
  final BackendApiService _api = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;
  final GroqService _groqService = GroqService();

  List<QuestionnaireQuestion> _questions = [];
  final Map<String, List<String>> _answers = {};
  int _currentIndex = 0;

  bool _isLoadingQuestions = true;
  bool _isSubmitting = false;
  String? _selectedDescription;

  @override
  void initState() {
    super.initState();
    _loadQuestionnaire();
  }

  Future<void> _loadQuestionnaire() async {
    setState(() {
      _isLoadingQuestions = true;
    });

    try {
      final loaded = await _api.fetchQuestionnaire();
      if (!mounted) {
        return;
      }

      setState(() {
        _questions = loaded;
        _isLoadingQuestions = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoadingQuestions = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to load questionnaire: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  QuestionnaireQuestion get _currentQuestion => _questions[_currentIndex];

  List<String> get _currentAnswer => _answers[_currentQuestion.id] ?? const [];

  bool get _isCurrentAnswered => _currentAnswer.isNotEmpty;

  double get _progress {
    if (_questions.isEmpty) {
      return 0;
    }
    return ((_currentIndex + 1) / _questions.length).clamp(0.0, 1.0);
  }

  void _onOptionTap(QuestionnaireOption option) {
    final question = _currentQuestion;
    final current = List<String>.from(_answers[question.id] ?? const []);

    if (question.isMultiSelect) {
      if (current.contains(option.id)) {
        current.remove(option.id);
      } else {
        current.add(option.id);
      }
    } else {
      current
        ..clear()
        ..add(option.id);
    }

    setState(() {
      _answers[question.id] = current;
      _selectedDescription = option.description;
    });
  }

  void _showDescription(QuestionnaireOption option) {
    setState(() {
      _selectedDescription = option.description ??
          'No additional description is available for this option.';
    });
  }

  Future<void> _nextOrSubmit() async {
    if (!_isCurrentAnswered) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please answer this question before continuing.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    if (_currentIndex < _questions.length - 1) {
      setState(() {
        _currentIndex += 1;
        _selectedDescription = null;
      });
      return;
    }

    await _submitAssessment();
  }

  Future<void> _submitAssessment() async {
    if (_isSubmitting) {
      return;
    }

    if (_answers.length != _questions.length) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please answer all questions before submitting.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    try {
      await _api.submitResponses(
        userId: widget.userId,
        selectedOptionIdsByQuestion: _answers,
      );

      final prediction = await _api.runPrediction(widget.userId);
      final localUser = await _localStorage.findByUserId(widget.userId);
      final llmReport = await _generateLlmReport(prediction, localUser);

      if (!mounted) {
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => HealthReportPage(
            userId: widget.userId,
            predictionData: prediction,
            localUser: localUser,
            reportText: llmReport,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to complete assessment: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  Future<String?> _generateLlmReport(
    Map<String, dynamic> prediction,
    LocalUserProfile? localUser,
  ) async {
    try {
      final raw = prediction['predictions'];
      if (raw is! Map<String, dynamic> || raw.isEmpty) {
        return null;
      }

      final ranked = raw.entries.map((entry) {
        final data = entry.value is Map<String, dynamic>
            ? entry.value as Map<String, dynamic>
            : <String, dynamic>{};
        final probability = (data['probability'] as num? ?? 0).toDouble();
        final label = data['label'] as String? ?? 'Unknown';
        return {
          'condition': entry.key,
          'probability': probability,
          'label': label,
        };
      }).toList()
        ..sort((a, b) => ((b['probability'] as double)
            .compareTo(a['probability'] as double)));

      final top = ranked.take(3).map((item) {
        final probability = (item['probability'] as double).toStringAsFixed(2);
        return '${item['condition']}: $probability (${item['label']})';
      }).join('; ');

      final userSnapshot = [
        if (localUser?.fullName != null) 'Name: ${localUser!.fullName}',
        if (localUser?.activityLevel != null)
          'Activity level: ${localUser!.activityLevel}',
        if (localUser?.maritalStatus != null)
          'Marital status: ${localUser!.maritalStatus}',
      ].join(', ');

      final prompt =
          '''Create a detailed women's health assessment explanation based on prediction output.
    Write 220-320 words in clear, supportive language.
    Use these exact section headers:
    Summary:
    What the risk scores mean:
    Possible contributing factors from current profile:
    Action plan for next 2 weeks:
    When to seek medical review:
    Medical disclaimer:
    Under "What the risk scores mean", explain top 3 risks with one line each.
    Avoid diagnosis, avoid fear-based language, and avoid mentioning model limitations.
    Top risk summary: $top.
    User context: ${userSnapshot.isEmpty ? 'Not available' : userSnapshot}.''';

      final result = await _groqService.sendSimpleMessage(prompt);
      final cleaned = result.trim();
      if (cleaned.isEmpty ||
          cleaned
              .startsWith('I apologize, but I\'m having trouble connecting')) {
        return _buildDetailedFallbackReport(ranked, localUser);
      }
      return cleaned;
    } catch (_) {
      return _buildDetailedFallbackReport(
          const <Map<String, dynamic>>[], localUser);
    }
  }

  String _buildDetailedFallbackReport(
    List<Map<String, dynamic>> ranked,
    LocalUserProfile? localUser,
  ) {
    final top = ranked.take(3).toList();
    String topLine(int index) {
      if (index >= top.length) {
        return '- Not enough scored conditions available yet.';
      }
      final item = top[index];
      final condition = item['condition'] as String? ?? 'Condition';
      final probability = (item['probability'] as num? ?? 0).toDouble();
      final label = item['label'] as String? ?? 'Unknown';
      return '- $condition: score ${probability.toStringAsFixed(2)} ($label). This indicates relative priority for monitoring, not a diagnosis.';
    }

    final profileFactors = <String>[
      if ((localUser?.activityLevel ?? '').isNotEmpty)
        'Activity level: ${localUser!.activityLevel}',
      if ((localUser?.maritalStatus ?? '').isNotEmpty)
        'Marital status: ${localUser!.maritalStatus}',
      if (localUser?.hasChronicConditions == true)
        'History of chronic conditions reported',
      if (localUser?.hasAllergies == true) 'Allergy history reported',
    ];

    return '''Summary:
Your assessment suggests a mixed risk profile, with some conditions needing closer follow-up than others. These scores help prioritize preventive actions and discussions with a clinician.

What the risk scores mean:
${topLine(0)}
${topLine(1)}
${topLine(2)}

Possible contributing factors from current profile:
${profileFactors.isEmpty ? '- Limited profile data is available, so this explanation is conservative.' : profileFactors.map((f) => '- $f').join('\n')}

Action plan for next 2 weeks:
- Track key symptoms daily (pain, cycle pattern, mood, fatigue) in one place.
- Maintain hydration, regular sleep, and consistent meals with fewer processed foods.
- Add light-to-moderate physical activity most days if comfortable.
- Bring this report and symptom trends to your next healthcare consultation.

When to seek medical review:
- Seek timely review if symptoms worsen, become persistent, or affect daily functioning.
- Seek urgent care for severe pain, heavy bleeding, dizziness, fainting, or new alarming symptoms.

Medical disclaimer:
This report is a screening-oriented interpretation and is not a medical diagnosis. Clinical examination and professional medical advice are required for decisions about tests or treatment.''';
  }

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
          child: _isLoadingQuestions
              ? const Center(
                  child: CircularProgressIndicator(color: Color(0xFFC85A7A)),
                )
              : _questions.isEmpty
                  ? _buildEmptyState()
                  : Column(
                      children: [
                        _buildHeader(),
                        Expanded(
                          child: SingleChildScrollView(
                            padding: const EdgeInsets.all(20),
                            child: Column(
                              children: [
                                _buildQuestionCard(),
                                const SizedBox(height: 20),
                                _buildOptionsList(),
                                const SizedBox(height: 16),
                                _buildDescriptionPanel(),
                                const SizedBox(height: 16),
                                _buildHelpfulTip(),
                              ],
                            ),
                          ),
                        ),
                        _buildBottomActions(),
                      ],
                    ),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.assignment_late, size: 72, color: Color(0xFFC85A7A)),
        const SizedBox(height: 12),
        const Text(
          'Questionnaire is not configured yet.',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 16),
        ElevatedButton(
          onPressed: _loadQuestionnaire,
          child: const Text('Retry'),
        ),
      ],
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFD4879C), Color(0xFFE5A1A1)],
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFD4879C).withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          )
        ],
      ),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.favorite, color: Colors.white, size: 28),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Question ${_currentIndex + 1} of ${_questions.length}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Text(
                '${(_progress * 100).round()}%',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: _progress,
              minHeight: 8,
              backgroundColor: Colors.white.withOpacity(0.3),
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuestionCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _currentQuestion.category,
            style: const TextStyle(
              color: Color(0xFFC85A7A),
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _currentQuestion.text,
            style: const TextStyle(
              fontSize: 19,
              fontWeight: FontWeight.bold,
              color: Color(0xFF2D2D2D),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _currentQuestion.isMultiSelect
                ? 'Select one or more options.'
                : 'Select one option.',
            style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildOptionsList() {
    return Column(
      children: _currentQuestion.options.map((option) {
        final selected = _currentAnswer.contains(option.id);
        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            color: selected ? const Color(0xFFFCE7F3) : Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color:
                  selected ? const Color(0xFFC85A7A) : const Color(0xFFF5D7E3),
              width: selected ? 2 : 1.4,
            ),
          ),
          child: InkWell(
            borderRadius: BorderRadius.circular(14),
            onTap: () => _onOptionTap(option),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              child: Row(
                children: [
                  Icon(
                    selected
                        ? Icons.check_circle
                        : (_currentQuestion.isMultiSelect
                            ? Icons.radio_button_unchecked
                            : Icons.circle_outlined),
                    color: selected ? const Color(0xFFC85A7A) : Colors.grey,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      option.text,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  IconButton(
                    tooltip: 'Option info',
                    onPressed: () => _showDescription(option),
                    icon: const Icon(
                      Icons.info_outline,
                      color: Color(0xFFC85A7A),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildDescriptionPanel() {
    final text = (_selectedDescription == null || _selectedDescription!.isEmpty)
        ? 'Tap an info icon to view option details here.'
        : _selectedDescription!;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF0F3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFF5D7E3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline, color: Color(0xFFC85A7A), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 13,
                color: Colors.grey.shade700,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHelpfulTip() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFF5D7E3)),
      ),
      child: const Row(
        children: [
          Icon(Icons.auto_awesome, color: Color(0xFFC85A7A)),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Answer based on your recent health experience for better model predictions.',
              style: TextStyle(fontSize: 13, color: Color(0xFF5D5D5D)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomActions() {
    final isLast = _currentIndex == _questions.length - 1;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
      child: Row(
        children: [
          if (_currentIndex > 0)
            Expanded(
              child: OutlinedButton(
                onPressed: _isSubmitting
                    ? null
                    : () {
                        setState(() {
                          _currentIndex -= 1;
                          _selectedDescription = null;
                        });
                      },
                child: const Text('Previous'),
              ),
            ),
          if (_currentIndex > 0) const SizedBox(width: 12),
          Expanded(
            flex: 2,
            child: ElevatedButton(
              onPressed: _isSubmitting ? null : _nextOrSubmit,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC85A7A),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: _isSubmitting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2.4,
                        color: Colors.white,
                      ),
                    )
                  : Text(isLast ? 'Submit & Generate Report' : 'Next Question'),
            ),
          ),
        ],
      ),
    );
  }
}
