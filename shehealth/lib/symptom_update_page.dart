import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'models/app_models.dart';
import 'report.dart';
import 'services/backend_api_service.dart';
import 'services/groq_service.dart';
import 'services/local_storage_service.dart';

class SymptomUpdatePage extends StatefulWidget {
  final String userId;

  const SymptomUpdatePage({super.key, required this.userId});

  @override
  State<SymptomUpdatePage> createState() => _SymptomUpdatePageState();
}

class _SymptomUpdatePageState extends State<SymptomUpdatePage> {
  final BackendApiService _api = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;
  final GroqService _groqService = GroqService();

  final Map<String, List<String>> _answers = {};
  final Map<String, List<String>> _initialAnswers = {};
  final Map<String, TextEditingController> _inputControllers = {};

  List<QuestionnaireQuestion> _questions = [];
  bool _isLoading = true;
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void dispose() {
    for (final controller in _inputControllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  bool _isInputQuestion(QuestionnaireQuestion q) => q.qType == 'input';

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final questions = await _api.fetchQuestionnaire();
      final loadedAnswers = <String, List<String>>{};

      try {
        final existing = await _api.getUserResponses(widget.userId);
        final rows = existing['responses'];
        if (rows is List) {
          for (final row in rows) {
            if (row is! Map<String, dynamic>) {
              continue;
            }
            final qid = row['question_id']?.toString();
            final selected = row['selected_option_ids'];
            if (qid == null || selected is! List) {
              continue;
            }

            loadedAnswers[qid] = selected
                .map((e) => e?.toString())
                .whereType<String>()
                .where((e) => e.isNotEmpty)
                .toList();
          }
        }
      } catch (_) {
        // First-time users may not have saved responses yet.
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _questions = questions;
        _answers
          ..clear()
          ..addAll(loadedAnswers.map(
            (key, value) => MapEntry(key, List<String>.from(value)),
          ));
        _initialAnswers
          ..clear()
          ..addAll(loadedAnswers.map(
            (key, value) => MapEntry(key, List<String>.from(value)),
          ));
        _isLoading = false;
      });

      for (final q in _questions.where(_isInputQuestion)) {
        _getControllerForQuestion(q);
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to load symptom form: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  TextEditingController _getControllerForQuestion(QuestionnaireQuestion q) {
    return _inputControllers.putIfAbsent(q.id, () {
      final stored = _decodeInputValue(_answers[q.id]);
      return TextEditingController(text: stored);
    });
  }

  String _decodeInputValue(List<String>? selected) {
    if (selected == null || selected.isEmpty) {
      return '';
    }
    final raw = selected.first;
    if (raw.startsWith('INPUT::')) {
      return raw.substring('INPUT::'.length);
    }
    return raw;
  }

  void _onInputChanged(QuestionnaireQuestion q, String raw) {
    final value = raw.trim();
    setState(() {
      if (value.isEmpty) {
        _answers.remove(q.id);
      } else {
        _answers[q.id] = ['INPUT::$value'];
      }
    });
  }

  void _onOptionTap(QuestionnaireQuestion q, QuestionnaireOption option) {
    final current = List<String>.from(_answers[q.id] ?? const []);

    if (q.isMultiSelect) {
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
      if (current.isEmpty) {
        _answers.remove(q.id);
      } else {
        _answers[q.id] = current;
      }
    });
  }

  Map<String, List<String>> _getChangedAnswers() {
    final changed = <String, List<String>>{};

    for (final q in _questions) {
      final before = _initialAnswers[q.id] ?? const [];
      final now = _answers[q.id] ?? const [];
      if (!listEquals(before, now) && now.isNotEmpty) {
        changed[q.id] = List<String>.from(now);
      }
    }

    return changed;
  }

  Future<void> _saveAndGenerateReport() async {
    if (_isSubmitting) {
      return;
    }

    final changed = _getChangedAnswers();
    if (changed.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No changes detected. Update at least one response.'),
          backgroundColor: Color(0xFFC85A7A),
        ),
      );
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    try {
      await _api.updateResponses(
        userId: widget.userId,
        selectedOptionIdsByQuestion: changed,
      );

      final currentPrediction = await _api.runPrediction(widget.userId);
      final comparison = currentPrediction['comparison'];
      final localUser = await _localStorage.findByUserId(widget.userId);
      final llmReport = await _generateLlmReport(
        currentPrediction,
        localUser,
        comparison: comparison is Map<String, dynamic> ? comparison : null,
      );

      if (!mounted) {
        return;
      }

      final comparisonMessage = _buildComparisonMessage(comparison);
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
          builder: (_) => HealthReportPage(
            userId: widget.userId,
            predictionData: currentPrediction,
            localUser: localUser,
            reportText: llmReport,
          ),
        ),
      );

      _initialAnswers
        ..clear()
        ..addAll(_answers.map((k, v) => MapEntry(k, List<String>.from(v))));
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to update symptoms: $e'),
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

  String? _buildComparisonMessage(dynamic comparisonRaw) {
    if (comparisonRaw is! Map<String, dynamic>) {
      return null;
    }

    final type = comparisonRaw['change_type']?.toString();
    if (type == 'drastic') {
      return 'Risk level changed significantly. Previous and current predictions are both kept.';
    }
    if (type == 'slight') {
      return 'Only slight change detected. Latest prediction replaced previous one.';
    }
    return null;
  }

  Future<String?> _generateLlmReport(
    Map<String, dynamic> prediction,
    LocalUserProfile? localUser, {
    Map<String, dynamic>? comparison,
  }) async {
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

      final trendHint = comparison == null
          ? 'Trend information is not available.'
          : 'Change type: ${comparison['change_type'] ?? 'unknown'}.';

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
Avoid diagnosis and fear-based language.
Top risk summary: $top.
Trend context: $trendHint.
User context: ${localUser?.fullName ?? 'Not available'}.''';

      final result = await _groqService.sendSimpleMessage(prompt);
      final cleaned = result.trim();
      if (cleaned.isEmpty ||
          cleaned.startsWith('I apologize, but I\'m having trouble connecting')) {
        return null;
      }
      return cleaned;
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Update Symptoms'),
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Color(0xFFC85A7A)),
            )
          : Column(
              children: [
                Container(
                  width: double.infinity,
                  margin: const EdgeInsets.all(16),
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF0F3),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFF5D7E3)),
                  ),
                  child: const Text(
                    'All questions are shown below. Existing values are pre-filled. Update only the changes you noticed and save.',
                    style: TextStyle(fontSize: 13, color: Color(0xFF5D5D5D)),
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    itemCount: _questions.length,
                    itemBuilder: (context, index) {
                      final q = _questions[index];
                      return _buildQuestionCard(index, q);
                    },
                  ),
                ),
              ],
            ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
          child: ElevatedButton(
            onPressed: _isSubmitting ? null : _saveAndGenerateReport,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              foregroundColor: Colors.white,
              minimumSize: const Size(double.infinity, 52),
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
                : const Text('Save Changes and Generate Report'),
          ),
        ),
      ),
    );
  }

  Widget _buildQuestionCard(int index, QuestionnaireQuestion q) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFF5D7E3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${index + 1}. ${q.text}',
            style: const TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 15,
              color: Color(0xFF2D2D2D),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            '${q.category.toUpperCase()} | ${q.qType}',
            style: const TextStyle(
              fontSize: 11,
              color: Color(0xFFC85A7A),
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          if (_isInputQuestion(q))
            TextField(
              controller: _getControllerForQuestion(q),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              onChanged: (value) => _onInputChanged(q, value),
              decoration: const InputDecoration(
                labelText: 'Enter value',
                hintText: 'Numeric input',
                prefixIcon: Icon(Icons.edit_outlined),
              ),
            )
          else
            Column(
              children: q.options.map((option) {
                final selected = (_answers[q.id] ?? const []).contains(option.id);
                return Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: selected ? const Color(0xFFFCE7F3) : Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: selected
                          ? const Color(0xFFC85A7A)
                          : const Color(0xFFF5D7E3),
                    ),
                  ),
                  child: ListTile(
                    dense: true,
                    onTap: () => _onOptionTap(q, option),
                    leading: Icon(
                      q.isMultiSelect
                          ? (selected
                              ? Icons.check_box
                              : Icons.check_box_outline_blank)
                          : (selected
                              ? Icons.radio_button_checked
                              : Icons.radio_button_unchecked),
                      color: selected ? const Color(0xFFC85A7A) : Colors.grey,
                    ),
                    title: Text(option.text),
                    subtitle: (option.description == null ||
                            option.description!.trim().isEmpty)
                        ? null
                        : Text(option.description!),
                  ),
                );
              }).toList(),
            ),
        ],
      ),
    );
  }
}
