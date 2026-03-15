import 'package:flutter/material.dart';

import 'models/app_models.dart';
import 'questionnaire.dart';
import 'services/backend_api_service.dart';
import 'services/local_storage_service.dart';
import 'services/session_service.dart';

class PersonalDetailsPage extends StatefulWidget {
  final String userId;
  final String fullName;
  final String email;
  final String phone;
  final String password;

  const PersonalDetailsPage({
    super.key,
    required this.userId,
    required this.fullName,
    required this.email,
    required this.phone,
    required this.password,
  });

  @override
  State<PersonalDetailsPage> createState() => _PersonalDetailsPageState();
}

class _PersonalDetailsPageState extends State<PersonalDetailsPage> {
  final _formKey = GlobalKey<FormState>();

  // Controllers
  final TextEditingController _dobController = TextEditingController();
  final TextEditingController _weightController = TextEditingController();
  final TextEditingController _heightController = TextEditingController();
  final TextEditingController _emergencyContactController =
      TextEditingController();

  // Dropdown selections
  String? _selectedBloodGroup;
  String? _selectedMaritalStatus;
  String? _selectedActivityLevel;

  // Toggle selections
  bool _hasAllergies = false;
  bool _hasChronicConditions = false;
  bool _isOnMedication = false;

  int _currentStep = 0;
  bool _isSaving = false;

  final BackendApiService _backendApi = BackendApiService();
  final LocalStorageService _localStorage = LocalStorageService.instance;
  final SessionService _sessionService = SessionService();

  final List<String> _bloodGroups = [
    'A+',
    'A-',
    'B+',
    'B-',
    'AB+',
    'AB-',
    'O+',
    'O-'
  ];
  final List<String> _maritalStatuses = [
    'Single',
    'Married',
    'Divorced',
    'Widowed',
    'Prefer not to say'
  ];
  final List<String> _activityLevels = [
    'Sedentary',
    'Lightly Active',
    'Moderately Active',
    'Very Active'
  ];

  @override
  void dispose() {
    _dobController.dispose();
    _weightController.dispose();
    _heightController.dispose();
    _emergencyContactController.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime(2000, 1, 1),
      firstDate: DateTime(1950),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: Color(0xFFC85A7A),
              onPrimary: Colors.white,
              surface: Colors.white,
              onSurface: Colors.black87,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _dobController.text =
            '${picked.day.toString().padLeft(2, '0')}/${picked.month.toString().padLeft(2, '0')}/${picked.year}';
      });
    }
  }

  void _nextStep() {
    if (_currentStep < 2) {
      setState(() => _currentStep++);
    } else {
      _submitDetails();
    }
  }

  void _prevStep() {
    if (_currentStep > 0) {
      setState(() => _currentStep--);
    }
  }

  Future<void> _submitDetails() async {
    if (_isSaving) {
      return;
    }

    final dob = _dobController.text.trim();
    final height = double.tryParse(_heightController.text.trim());
    final weight = double.tryParse(_weightController.text.trim());

    if (dob.isEmpty ||
        height == null ||
        weight == null ||
        _selectedMaritalStatus == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please complete date of birth, height, weight, and marital status.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    final age = _calculateAgeFromDob(dob);
    if (age == null || age <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please provide a valid date of birth.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() => _isSaving = true);

    try {
      final localUser = LocalUserProfile(
        userId: widget.userId,
        fullName: widget.fullName,
        email: widget.email,
        phone: widget.phone,
        password: widget.password,
        dob: dob,
        bloodGroup: _selectedBloodGroup,
        maritalStatus: _selectedMaritalStatus,
        activityLevel: _selectedActivityLevel,
        emergencyContact: _emergencyContactController.text.trim(),
        hasAllergies: _hasAllergies,
        hasChronicConditions: _hasChronicConditions,
        isOnMedication: _isOnMedication,
        heightCm: height,
        weightKg: weight,
      );

      await _localStorage.upsertUser(localUser);

      // Persist only backend-approved user profile fields in MongoDB.
      await _backendApi.registerUserProfile(
        userId: widget.userId,
        age: age,
        height: height,
        weight: weight,
        maritalStatus: _selectedMaritalStatus,
        familyHistory: _hasChronicConditions,
      );

      await _sessionService.setCurrentUserId(widget.userId);

      if (!mounted) {
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => SymptomQuestionnaire(userId: widget.userId),
        ),
      );
    } catch (e) {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to save profile: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
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
      final hasBirthdayPassed =
          now.month > birthDate.month || (now.month == birthDate.month && now.day >= birthDate.day);
      if (!hasBirthdayPassed) {
        age -= 1;
      }
      return age;
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFDF2F8),
      body: Column(
        children: [
          _buildHeader(),
          _buildStepIndicator(),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  _buildStepContent(),
                  const SizedBox(height: 24),
                  _buildNavigationButtons(),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 55, 20, 24),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFFC85A7A), Color(0xFFE59393), Color(0xFFFFE1E1)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.25),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.arrow_back_ios_new,
                  color: Colors.white, size: 18),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Hi, ${widget.fullName.split(' ').first}! 💗',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 26,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Tell us a bit more about yourself so we can personalize your health experience.',
            style: TextStyle(color: Colors.white70, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildStepIndicator() {
    final steps = ['Basic Info', 'Body Metrics', 'Health History'];
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
      child: Row(
        children: List.generate(steps.length, (i) {
          final isDone = i < _currentStep;
          final isActive = i == _currentStep;
          return Expanded(
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    children: [
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: isDone
                              ? const Color(0xFF4CAF50)
                              : isActive
                                  ? const Color(0xFFC85A7A)
                                  : const Color(0xFFFCE7F3),
                          shape: BoxShape.circle,
                        ),
                        child: Center(
                          child: isDone
                              ? const Icon(Icons.check,
                                  color: Colors.white, size: 16)
                              : Text(
                                  '${i + 1}',
                                  style: TextStyle(
                                    color: isActive
                                        ? Colors.white
                                        : const Color(0xFFE59393),
                                    fontWeight: FontWeight.bold,
                                    fontSize: 13,
                                  ),
                                ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        steps[i],
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight:
                              isActive ? FontWeight.bold : FontWeight.normal,
                          color:
                              isActive ? const Color(0xFFC85A7A) : Colors.grey,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
                if (i < steps.length - 1)
                  Expanded(
                    child: Container(
                      height: 2,
                      margin: const EdgeInsets.only(bottom: 24),
                      color: i < _currentStep
                          ? const Color(0xFF4CAF50)
                          : const Color(0xFFFCE7F3),
                    ),
                  ),
              ],
            ),
          );
        }),
      ),
    );
  }

  Widget _buildStepContent() {
    switch (_currentStep) {
      case 0:
        return _buildBasicInfoStep();
      case 1:
        return _buildBodyMetricsStep();
      case 2:
        return _buildHealthHistoryStep();
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _buildBasicInfoStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionTitle('Basic Information', Icons.person_outline),
        const SizedBox(height: 16),
        _buildReadOnlyField('Full Name', widget.fullName, Icons.person),
        const SizedBox(height: 14),
        _buildReadOnlyField(
            'Email Address', widget.email, Icons.email_outlined),
        const SizedBox(height: 14),
        _buildDateField(),
        const SizedBox(height: 14),
        _buildDropdownField(
          label: 'Blood Group',
          icon: Icons.bloodtype_outlined,
          value: _selectedBloodGroup,
          items: _bloodGroups,
          onChanged: (val) => setState(() => _selectedBloodGroup = val),
        ),
        const SizedBox(height: 14),
        _buildDropdownField(
          label: 'Marital Status',
          icon: Icons.favorite_border,
          value: _selectedMaritalStatus,
          items: _maritalStatuses,
          onChanged: (val) => setState(() => _selectedMaritalStatus = val),
        ),
        const SizedBox(height: 14),
        _buildInputField(
          controller: _emergencyContactController,
          label: 'Emergency Contact Number',
          icon: Icons.phone_outlined,
          keyboardType: TextInputType.phone,
        ),
      ],
    );
  }

  Widget _buildBodyMetricsStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionTitle('Body Metrics', Icons.monitor_weight_outlined),
        const SizedBox(height: 16),
        _buildInputField(
          controller: _weightController,
          label: 'Weight (kg)',
          icon: Icons.monitor_weight_outlined,
          keyboardType: TextInputType.number,
        ),
        const SizedBox(height: 14),
        _buildInputField(
          controller: _heightController,
          label: 'Height (cm)',
          icon: Icons.height,
          keyboardType: TextInputType.number,
        ),
        const SizedBox(height: 14),
        _buildDropdownField(
          label: 'Activity Level',
          icon: Icons.directions_run_outlined,
          value: _selectedActivityLevel,
          items: _activityLevels,
          onChanged: (val) => setState(() => _selectedActivityLevel = val),
        ),
        const SizedBox(height: 20),
        // BMI Preview Card
        if (_weightController.text.isNotEmpty &&
            _heightController.text.isNotEmpty)
          _buildBmiCard(),
      ],
    );
  }

  Widget _buildBmiCard() {
    final weight = double.tryParse(_weightController.text) ?? 0;
    final heightCm = double.tryParse(_heightController.text) ?? 0;
    if (weight <= 0 || heightCm <= 0) return const SizedBox.shrink();
    final heightM = heightCm / 100;
    final bmi = weight / (heightM * heightM);
    String bmiCategory;
    Color bmiColor;
    if (bmi < 18.5) {
      bmiCategory = 'Underweight';
      bmiColor = Colors.blue;
    } else if (bmi < 25) {
      bmiCategory = 'Normal weight';
      bmiColor = Colors.green;
    } else if (bmi < 30) {
      bmiCategory = 'Overweight';
      bmiColor = Colors.orange;
    } else {
      bmiCategory = 'Obese';
      bmiColor = Colors.red;
    }
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bmiColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: bmiColor.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: bmiColor.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.insights, color: bmiColor, size: 24),
          ),
          const SizedBox(width: 14),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Your BMI',
                  style: TextStyle(color: Colors.grey, fontSize: 12)),
              Text(
                bmi.toStringAsFixed(1),
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold, color: bmiColor),
              ),
              Text(bmiCategory,
                  style: TextStyle(
                      color: bmiColor,
                      fontSize: 13,
                      fontWeight: FontWeight.w600)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHealthHistoryStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionTitle(
            'Health History', Icons.medical_information_outlined),
        const SizedBox(height: 16),
        _buildToggleCard(
          title: 'Do you have any allergies?',
          subtitle: 'Food, medication, environmental, etc.',
          icon: Icons.warning_amber_outlined,
          color: Colors.orange,
          value: _hasAllergies,
          onChanged: (val) => setState(() => _hasAllergies = val),
        ),
        const SizedBox(height: 12),
        _buildToggleCard(
          title: 'Any chronic conditions?',
          subtitle: 'Diabetes, hypertension, thyroid, etc.',
          icon: Icons.monitor_heart_outlined,
          color: Colors.red,
          value: _hasChronicConditions,
          onChanged: (val) => setState(() => _hasChronicConditions = val),
        ),
        const SizedBox(height: 12),
        _buildToggleCard(
          title: 'Currently on any medication?',
          subtitle: 'Prescription drugs, supplements, etc.',
          icon: Icons.medication_outlined,
          color: const Color(0xFFC85A7A),
          value: _isOnMedication,
          onChanged: (val) => setState(() => _isOnMedication = val),
        ),
        const SizedBox(height: 20),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFFFCE7F3),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFFFBCFE8)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.info_outline,
                  color: Color(0xFFC85A7A), size: 20),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Your health information is private and securely stored. It helps us predict health risks and give you personalized recommendations.',
                  style: TextStyle(
                      fontSize: 12, color: Colors.grey.shade700, height: 1.5),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [Color(0xFFC85A7A), Color(0xFFE59393)]),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
        const SizedBox(width: 12),
        Text(
          title,
          style: const TextStyle(
              fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
        ),
      ],
    );
  }

  Widget _buildReadOnlyField(String label, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Row(
        children: [
          Icon(icon, color: const Color(0xFFE59393), size: 22),
          const SizedBox(width: 14),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label,
                  style: const TextStyle(color: Colors.grey, fontSize: 11)),
              const SizedBox(height: 2),
              Text(value,
                  style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: Colors.black87)),
            ],
          ),
          const Spacer(),
          const Icon(Icons.lock_outline, color: Colors.grey, size: 16),
        ],
      ),
    );
  }

  Widget _buildDateField() {
    return GestureDetector(
      onTap: _pickDate,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
        ),
        child: Row(
          children: [
            const Icon(Icons.cake_outlined, color: Color(0xFFE59393), size: 22),
            const SizedBox(width: 14),
            Text(
              _dobController.text.isEmpty
                  ? 'Date of Birth'
                  : _dobController.text,
              style: TextStyle(
                fontSize: 15,
                color: _dobController.text.isEmpty
                    ? Colors.grey.shade500
                    : Colors.black87,
              ),
            ),
            const Spacer(),
            const Icon(Icons.calendar_today_outlined,
                color: Color(0xFFE59393), size: 18),
          ],
        ),
      ),
    );
  }

  Widget _buildInputField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: keyboardType,
      onChanged: (_) => setState(() {}),
      decoration: InputDecoration(
        hintText: label,
        hintStyle: TextStyle(color: Colors.grey.shade500, fontSize: 15),
        prefixIcon: Icon(icon, color: const Color(0xFFE59393), size: 22),
        filled: true,
        fillColor: Colors.white,
        contentPadding:
            const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: Color(0xFFFCE7F3), width: 1.5),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: Color(0xFFC85A7A), width: 1.5),
        ),
      ),
    );
  }

  Widget _buildDropdownField({
    required String label,
    required IconData icon,
    required String? value,
    required List<String> items,
    required void Function(String?) onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          isExpanded: true,
          hint: Row(
            children: [
              Icon(icon, color: const Color(0xFFE59393), size: 22),
              const SizedBox(width: 14),
              Text(label,
                  style: TextStyle(color: Colors.grey.shade500, fontSize: 15)),
            ],
          ),
          icon: const Icon(Icons.keyboard_arrow_down, color: Color(0xFFE59393)),
          items: items
              .map((item) => DropdownMenuItem(value: item, child: Text(item)))
              .toList(),
          onChanged: onChanged,
          selectedItemBuilder: (context) => items
              .map((item) => Row(
                    children: [
                      Icon(icon, color: const Color(0xFFE59393), size: 22),
                      const SizedBox(width: 14),
                      Text(item,
                          style: const TextStyle(
                              fontSize: 15, color: Colors.black87)),
                    ],
                  ))
              .toList(),
        ),
      ),
    );
  }

  Widget _buildToggleCard({
    required String title,
    required String subtitle,
    required IconData icon,
    required Color color,
    required bool value,
    required void Function(bool) onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: value ? color.withOpacity(0.08) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: value ? color.withOpacity(0.4) : const Color(0xFFFCE7F3),
          width: 1.5,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        fontWeight: FontWeight.w600, fontSize: 14)),
                const SizedBox(height: 2),
                Text(subtitle,
                    style:
                        TextStyle(color: Colors.grey.shade500, fontSize: 12)),
              ],
            ),
          ),
          Switch(
            value: value,
            onChanged: onChanged,
            activeColor: color,
            activeTrackColor: color.withOpacity(0.3),
          ),
        ],
      ),
    );
  }

  Widget _buildNavigationButtons() {
    final isLastStep = _currentStep == 2;
    return Row(
      children: [
        if (_currentStep > 0) ...[
          Expanded(
            child: OutlinedButton(
              onPressed: _prevStep,
              style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFFC85A7A),
                side: const BorderSide(color: Color(0xFFC85A7A), width: 1.5),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16)),
              ),
              child: const Text('Back',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ),
          ),
          const SizedBox(width: 14),
        ],
        Expanded(
          flex: 2,
          child: ElevatedButton(
            onPressed: _isSaving ? null : _nextStep,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16)),
              elevation: 4,
              shadowColor: const Color(0xFFE59393).withOpacity(0.5),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  _isSaving
                      ? 'Saving...'
                      : isLastStep
                          ? 'Continue to Health Survey'
                          : 'Continue',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(width: 8),
                if (!_isSaving)
                  Icon(
                    isLastStep
                        ? Icons.assignment_outlined
                        : Icons.arrow_forward_rounded,
                    color: Colors.white,
                    size: 20,
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
