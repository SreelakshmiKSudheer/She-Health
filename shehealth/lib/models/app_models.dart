class LocalUserProfile {
  final String userId;
  final String fullName;
  final String email;
  final String phone;
  final String password;
  final String? dob;
  final String? bloodGroup;
  final String? maritalStatus;
  final String? activityLevel;
  final String? emergencyContact;
  final bool hasAllergies;
  final bool hasChronicConditions;
  final bool isOnMedication;
  final double? heightCm;
  final double? weightKg;

  const LocalUserProfile({
    required this.userId,
    required this.fullName,
    required this.email,
    required this.phone,
    required this.password,
    this.dob,
    this.bloodGroup,
    this.maritalStatus,
    this.activityLevel,
    this.emergencyContact,
    this.hasAllergies = false,
    this.hasChronicConditions = false,
    this.isOnMedication = false,
    this.heightCm,
    this.weightKg,
  });

  Map<String, Object?> toMap() {
    return {
      'user_id': userId,
      'full_name': fullName,
      'email': email,
      'phone': phone,
      'password': password,
      'dob': dob,
      'blood_group': bloodGroup,
      'marital_status': maritalStatus,
      'activity_level': activityLevel,
      'emergency_contact': emergencyContact,
      'has_allergies': hasAllergies ? 1 : 0,
      'has_chronic_conditions': hasChronicConditions ? 1 : 0,
      'is_on_medication': isOnMedication ? 1 : 0,
      'height_cm': heightCm,
      'weight_kg': weightKg,
    };
  }

  factory LocalUserProfile.fromMap(Map<String, Object?> map) {
    return LocalUserProfile(
      userId: map['user_id'] as String,
      fullName: map['full_name'] as String,
      email: map['email'] as String,
      phone: map['phone'] as String,
      password: map['password'] as String,
      dob: map['dob'] as String?,
      bloodGroup: map['blood_group'] as String?,
      maritalStatus: map['marital_status'] as String?,
      activityLevel: map['activity_level'] as String?,
      emergencyContact: map['emergency_contact'] as String?,
      hasAllergies: (map['has_allergies'] as int? ?? 0) == 1,
      hasChronicConditions: (map['has_chronic_conditions'] as int? ?? 0) == 1,
      isOnMedication: (map['is_on_medication'] as int? ?? 0) == 1,
      heightCm: (map['height_cm'] as num?)?.toDouble(),
      weightKg: (map['weight_kg'] as num?)?.toDouble(),
    );
  }
}

class QuestionnaireOption {
  final String id;
  final String text;
  final String? description;

  const QuestionnaireOption({
    required this.id,
    required this.text,
    this.description,
  });

  factory QuestionnaireOption.fromJson(Map<String, dynamic> json) {
    return QuestionnaireOption(
      id: json['id'] as String,
      text: json['text'] as String,
      description: json['description'] as String?,
    );
  }
}

class QuestionnaireQuestion {
  final String id;
  final String text;
  final String category;
  final String qType;
  final bool isInitial;
  final int priority;
  final List<QuestionnaireOption> options;

  const QuestionnaireQuestion({
    required this.id,
    required this.text,
    required this.category,
    required this.qType,
    required this.isInitial,
    required this.priority,
    required this.options,
  });

  bool get isMultiSelect => qType == 'multi_select';

  factory QuestionnaireQuestion.fromJson(Map<String, dynamic> json) {
    final options = (json['options'] as List<dynamic>? ?? [])
        .map((e) => QuestionnaireOption.fromJson(e as Map<String, dynamic>))
        .toList();

    return QuestionnaireQuestion(
      id: json['id'] as String,
      text: json['text'] as String,
      category: json['category'] as String,
      qType: json['q_type'] as String,
      isInitial: json['is_initial'] as bool? ?? true,
      priority: json['priority'] as int? ?? 0,
      options: options,
    );
  }
}
