import 'package:flutter/material.dart';

import 'models/app_models.dart';
import 'services/backend_api_service.dart';
import 'services/local_storage_service.dart';
import 'thyroid_report.dart';

class ThyroidQuestionnairePage extends StatefulWidget {
  final String userId;

  const ThyroidQuestionnairePage({super.key, required this.userId});

  @override
  State<ThyroidQuestionnairePage> createState() => _ThyroidQuestionnairePageState();
}

class _ThyroidQuestionnairePageState extends State<ThyroidQuestionnairePage> {
  final BackendApiService _api = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;

  List<QuestionnaireQuestion> _questions = [];
  final Map<String, List<String>> _answers = {};
  final Map<String, TextEditingController> _inputControllers = {};
  int _currentIndex = 0;
  bool _isLoading = true;
  bool _isSubmitting = false;
  String? _selectedDescription;

  @override
  void initState() {
    super.initState();
    _loadQuestions();
  }

  @override
  void dispose() {
    for (final controller in _inputControllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  bool get _isInputQuestion => _currentQuestion.qType == 'input';

  QuestionnaireQuestion get _currentQuestion => _questions[_currentIndex];

  List<String> get _currentAnswer => _answers[_currentQuestion.id] ?? const [];

  bool get _isCurrentAnswered {
    if (_isInputQuestion) {
      return _getInputController(_currentQuestion).text.trim().isNotEmpty;
    }
    return _currentAnswer.isNotEmpty;
  }

  double get _progress {
    if (_questions.isEmpty) {
      return 0;
    }
    return ((_currentIndex + 1) / _questions.length).clamp(0.0, 1.0);
  }

  Future<void> _loadQuestions() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final loaded = await _api.fetchThyroidQuestions();
      if (!mounted) {
        return;
      }

      setState(() {
        _questions = loaded;
        _isLoading = false;
      });

      for (final question in loaded.where((question) => question.qType == 'input')) {
        _getInputController(question);
      }
    } catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to load thyroid questions: $error'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  TextEditingController _getInputController(QuestionnaireQuestion question) {
    return _inputControllers.putIfAbsent(question.id, () {
      final existing = _decodeInputAnswer(_answers[question.id]);
      return TextEditingController(text: existing);
    });
  }

  String _decodeInputAnswer(List<String>? answer) {
    if (answer == null || answer.isEmpty) {
      return '';
    }

    final raw = answer.first;
    if (raw.startsWith('INPUT::')) {
      return raw.substring('INPUT::'.length);
    }
    return raw;
  }

  void _onInputChanged(String value) {
    final trimmed = value.trim();
    setState(() {
      if (trimmed.isEmpty) {
        _answers.remove(_currentQuestion.id);
      } else {
        _answers[_currentQuestion.id] = ['INPUT::$trimmed'];
      }
      _selectedDescription = null;
    });
  }

  void _onOptionTap(QuestionnaireOption option) {
    final question = _currentQuestion;
    final current = List<String>.from(_answers[question.id] ?? const []);

    bool isNoneOption(QuestionnaireOption opt) {
      final text = opt.text.trim().toLowerCase();
      return text == 'none' || text == 'no' || text == 'none of the above';
    }

    if (question.isMultiSelect) {
      final tappedIsNone = isNoneOption(option);

      if (tappedIsNone) {
        if (current.contains(option.id)) {
          current.remove(option.id);
        } else {
          current
            ..clear()
            ..add(option.id);
        }
      } else {
        final noneIds = question.options.where(isNoneOption).map((opt) => opt.id).toSet();
        current.removeWhere((id) => noneIds.contains(id));

        if (current.contains(option.id)) {
          current.remove(option.id);
        } else {
          current.add(option.id);
        }
      }
    } else {
      current
        ..clear()
        ..add(option.id);
    }

    setState(() {
      _answers[question.id] = current;
      _selectedDescription = option.description?.trim().isNotEmpty == true
          ? option.description
          : null;
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
          content: Text('Please answer all thyroid questions before submitting.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    try {
      await _api.submitThyroidResponses(
        userId: widget.userId,
        selectedOptionIdsByQuestion: _answers,
      );

      final prediction = await _api.getThyroidPrediction(widget.userId);
      final localUser = await _localStorage.findByUserId(widget.userId);

      if (!mounted) {
        return;
      }

      final comparison = prediction['comparison'];
      final comparisonMessage = _comparisonMessage(comparison);
      if (comparisonMessage != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(comparisonMessage),
            backgroundColor: const Color(0xFFC85A7A),
          ),
        );
      }

      await Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ThyroidReportPage(
            userId: widget.userId,
            predictionData: prediction,
            localUser: localUser,
            reportText: null,
          ),
        ),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to complete thyroid assessment: $error'),
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

  String? _comparisonMessage(dynamic comparisonRaw) {
    if (comparisonRaw is! Map<String, dynamic>) {
      return null;
    }

    final changeType = comparisonRaw['change_type']?.toString();
    if (changeType == 'drastic') {
      return 'Thyroid risk changed significantly since the last assessment.';
    }
    if (changeType == 'slight') {
      return 'Only a slight thyroid risk change was detected.';
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    const pink = Color(0xFFC85A7A);
    const pinkSoft = Color(0xFFE59393);
    const pinkLight = Color(0xFFFFF5F8);

    return Scaffold(
      backgroundColor: pinkLight,
      appBar: AppBar(
        backgroundColor: pink,
        foregroundColor: Colors.white,
        title: const Text('Thyroid Assessment'),
        elevation: 0,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: pink))
          : _questions.isEmpty
              ? const Center(child: Text('No thyroid questions available.'))
              : SafeArea(
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'Question ${_currentIndex + 1} of ${_questions.length}',
                                  style: const TextStyle(
                                    color: Color(0xFF7B2D4E),
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Text(
                                  '${(_progress * 100).round()}%',
                                  style: const TextStyle(
                                    color: Color(0xFF7B2D4E),
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(999),
                              child: LinearProgressIndicator(
                                value: _progress,
                                minHeight: 10,
                                backgroundColor: const Color(0xFFF3D8E0),
                                valueColor: const AlwaysStoppedAnimation<Color>(pinkSoft),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 18),
                      Expanded(
                        child: SingleChildScrollView(
                          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: double.infinity,
                                padding: const EdgeInsets.all(22),
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(24),
                                  border: Border.all(color: const Color(0xFFF6D5E0)),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.04),
                                      blurRadius: 18,
                                      offset: const Offset(0, 10),
                                    ),
                                  ],
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                      decoration: BoxDecoration(
                                        color: pink.withOpacity(0.08),
                                        borderRadius: BorderRadius.circular(999),
                                      ),
                                      child: Text(
                                        _currentQuestion.category,
                                        style: const TextStyle(
                                          color: pink,
                                          fontSize: 12,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(height: 14),
                                    Text(
                                      _currentQuestion.text,
                                      style: const TextStyle(
                                        color: Color(0xFF4A2032),
                                        fontSize: 22,
                                        height: 1.2,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                    const SizedBox(height: 14),
                                    Text(
                                      _isInputQuestion
                                          ? 'Enter the value carefully. You can move back and forth before submitting.'
                                          : _currentQuestion.isMultiSelect
                                              ? 'Select all options that apply.'
                                              : 'Select the option that best matches your situation.',
                                      style: const TextStyle(
                                        color: Color(0xFF7A5C6A),
                                        fontSize: 13,
                                        height: 1.45,
                                      ),
                                    ),
                                    const SizedBox(height: 18),
                                    if (_isInputQuestion)
                                      _buildInputField()
                                    else
                                      _buildOptionsList(),
                                    if (_selectedDescription != null) ...[
                                      const SizedBox(height: 16),
                                      Container(
                                        width: double.infinity,
                                        padding: const EdgeInsets.all(14),
                                        decoration: BoxDecoration(
                                          color: const Color(0xFFFFF5F8),
                                          borderRadius: BorderRadius.circular(18),
                                          border: Border.all(color: const Color(0xFFF9D3DE)),
                                        ),
                                        child: Text(
                                          _selectedDescription!,
                                          style: const TextStyle(
                                            color: Color(0xFF7A5C6A),
                                            height: 1.4,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                        child: Row(
                          children: [
                            Expanded(
                              child: OutlinedButton(
                                onPressed: _currentIndex == 0
                                    ? null
                                    : () {
                                        setState(() {
                                          _currentIndex -= 1;
                                          _selectedDescription = null;
                                        });
                                      },
                                style: OutlinedButton.styleFrom(
                                  foregroundColor: pink,
                                  side: const BorderSide(color: pink),
                                  padding: const EdgeInsets.symmetric(vertical: 14),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                                ),
                                child: const Text('Previous'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: _isSubmitting ? null : _nextOrSubmit,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: pink,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.symmetric(vertical: 14),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                                ),
                                child: Text(_currentIndex == _questions.length - 1 ? 'Submit & Predict' : 'Next'),
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

  Widget _buildInputField() {
    final controller = _getInputController(_currentQuestion);
    return TextField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      onChanged: _onInputChanged,
      decoration: InputDecoration(
        hintText: 'Enter value',
        filled: true,
        fillColor: const Color(0xFFFFF7FA),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      ),
    );
  }

  Widget _buildOptionsList() {
    return Column(
      children: _currentQuestion.options.map((option) {
        final selected = _currentAnswer.contains(option.id);
        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: InkWell(
            onTap: () => _onOptionTap(option),
            borderRadius: BorderRadius.circular(18),
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: selected ? const Color(0xFFFFF0F5) : Colors.white,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: selected ? const Color(0xFFC85A7A) : const Color(0xFFF0DDE4)),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          option.text,
                          style: TextStyle(
                            color: selected ? const Color(0xFF8A244D) : const Color(0xFF4A2032),
                            fontWeight: FontWeight.w700,
                            fontSize: 15,
                          ),
                        ),
                        if ((option.description ?? '').trim().isNotEmpty) ...[
                          const SizedBox(height: 4),
                          Text(
                            option.description!,
                            style: const TextStyle(color: Color(0xFF7A5C6A), fontSize: 12, height: 1.35),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Icon(
                    _currentQuestion.isMultiSelect
                        ? (selected ? Icons.check_box : Icons.check_box_outline_blank)
                        : (selected ? Icons.radio_button_checked : Icons.radio_button_off),
                    color: selected ? const Color(0xFFC85A7A) : const Color(0xFFB38A98),
                  ),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}
