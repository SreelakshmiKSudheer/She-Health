import 'package:flutter/material.dart';
import 'services/groq_service.dart';
import 'services/session_service.dart';
import 'dart:convert';
import 'services/notification_service.dart';
import 'package:provider/provider.dart';
import 'state/app_state.dart';

class Meal {
  final String name;
  final String description;
  final String calories;
  final List<String> items;
  const Meal({required this.name, required this.description, required this.calories, required this.items});
}

class DayPlan {
  final String day;
  final Meal breakfast;
  final Meal lunch;
  final Meal dinner;
  final Meal snack;
 

  DayPlan({
  required this.day,
  required this.breakfast,
  required this.lunch,
  required this.dinner,
  required this.snack,
});

  factory DayPlan.fromJson(Map<String, dynamic> json) {
  Meal parseMeal(dynamic m) {
    return Meal(
      name: m['name'] ?? '',
      description: m['description'] ?? '',
      calories: m['calories']?.toString() ?? '0',
      items: (m['items'] is List)
          ? List<String>.from(m['items'])
          : [],
    );
  }

  return DayPlan(
    day: json['day'] ?? '',
    breakfast: parseMeal(json['breakfast']),
    lunch: parseMeal(json['lunch']),
    dinner: parseMeal(json['dinner']),
    snack: parseMeal(json['snack']),
  );
}
}

class Exercise {
  final String name;
  final String duration;
  final String intensity;
  final String benefit;
  final IconData icon;
  final List<String> steps;
  const Exercise({required this.name, required this.duration, required this.intensity, required this.benefit, required this.icon, required this.steps});
}

class WorkoutDay {
  final String day;
  final String focus;
  final List<Exercise> exercises;
  const WorkoutDay({required this.day, required this.focus, required this.exercises});
}

class DiseaseDietPlan {
  final String disease;
  final String subtitle;
  final String description;
  final IconData icon;
  final Color primaryColor;
  final Color accentColor;
  final List<String> keyNutrients;
  final List<String> foodsToAvoid;
  final List<String> superfoods;
  final List<DayPlan> weeklyPlan;
  final String exerciseOverview;
  final List<String> exerciseTips;
  final List<WorkoutDay> workoutWeek;
  const DiseaseDietPlan({
    required this.disease, required this.subtitle, required this.description,
    required this.icon, required this.primaryColor, required this.accentColor,
    required this.keyNutrients, required this.foodsToAvoid, required this.superfoods,
    required this.weeklyPlan, required this.exerciseOverview,
    required this.exerciseTips, required this.workoutWeek,
  });
}

class AiHealthPlanData {
  final String subtitle;
  final String description;
  final List<String> keyNutrients;
  final List<String> foodsToAvoid;
  final List<String> superfoods;
  final String exerciseOverview;
  final List<String> exerciseTips;

  const AiHealthPlanData({
    required this.subtitle,
    required this.description,
    required this.keyNutrients,
    required this.foodsToAvoid,
    required this.superfoods,
    required this.exerciseOverview,
    required this.exerciseTips,
  });
}

// Disease metadata for styling
final Map<String, Map<String, dynamic>>
    _diseaseMetadata = {

  'PCOS': {
    'icon': Icons.spa_outlined,
    'primaryColor': const Color(0xFFC85A7A),
    'accentColor': const Color(0xFFFFE4EC),
  },

  'Endometriosis': {
    'icon': Icons.favorite_border,
    'primaryColor': const Color(0xFF7C4D9F),
    'accentColor': const Color(0xFFF3E8FF),
  },

  'Cervical Cancer': {
    'icon': Icons.shield_outlined,
    'primaryColor': const Color(0xFF1E8A6E),
    'accentColor': const Color(0xFFDFF5EF),
  },
};

class DietPlanPage extends StatefulWidget {
  const DietPlanPage({Key? key}) : super(key: key);
  @override
  State<DietPlanPage> createState() => _DietPlanPageState();
}

class _DietPlanPageState extends State<DietPlanPage> with TickerProviderStateMixin {
  late TabController _tabController;
  int _selectedDayIndex = 0;
  int _selectedWorkoutDayIndex = 0;
  
  
  Map<String, DiseaseDietPlan> _generatedPlans = {};
  String _errorMessage = '';
  bool _isLoading = false;
  Map<String, bool> _notificationSent = {};
  
  final List<String> _diseases = ['PCOS', 'Endometriosis','Cervical Cancer'];
  String _currentDisease = 'PCOS';

  @override

  void initState() {
  super.initState();
  
  _triggerNotifications();
  _tabController = TabController(
    length: _diseases.length,
    vsync: this,
  );

  _tabController.addListener(_onDiseaseChanged);
  
  // Generate ONLY current tab
  _generateCurrentDiseasePlan();
}
void _triggerNotifications() {
  // 🔹 Fetch dynamic data (NO hardcoding)
  String diet = "Your diet plan for today!!";
  String workout = "Your workout plan for today!!";

  NotificationService.scheduleDietPlan(diet);
  NotificationService.scheduleWorkout(workout);
}
  void _onDiseaseChanged() {

  if (!_tabController.indexIsChanging) {

    setState(() {

      _currentDisease =
          _diseases[_tabController.index];

      _selectedDayIndex = 0;

      _selectedWorkoutDayIndex = 0;
    });

    _generateCurrentDiseasePlan();
  }
}
Future<void> _generateCurrentDiseasePlan() async {

  if (_isLoading) return;

  final disease = _currentDisease;

  // already generated
  if (_generatedPlans.containsKey(disease)) {
    return;
  }

  if (!mounted) return;

  setState(() {
    _isLoading = true;
    _errorMessage = '';
  });

  try {

    // =========================
    // METADATA
    // =========================

    AiHealthPlanData aiMetadata;

    try {

      aiMetadata =
          await _generateMetadataFromGroq(
              disease);

      print("✓ Metadata generated");

    } catch (e) {

      print("❌ Metadata failed");
      print(e);

      aiMetadata =
          const AiHealthPlanData(
        subtitle:
            'Personalized Health Plan',

        description:
            'AI wellness guidance.',

        keyNutrients: [],

        foodsToAvoid: [],

        superfoods: [],

        exerciseOverview:
            'Light exercise recommended.',

        exerciseTips: [],
      );
    }

    // =========================
    // MEALS
    // =========================

List<DayPlan> meals = [];

try {

  meals = await _generateWeeklyMealsSafe(disease);

  print("✓ Weekly meals generated (SAFE)");

} catch (e) {

  print("❌ Weekly meals failed");
  print(e);

  meals = [];
}

    // =========================
    // WORKOUTS
    // =========================

    List<WorkoutDay> workouts = [];

    try {

      workouts =
          await _generateWorkoutFromGroq(
              disease);

      print("✓ Workouts generated");

    } catch (e) {

      print("❌ Workouts failed");
      print(e);

      workouts = [];
    }
    if (_notificationSent[disease] != true) {

  int todayIndex = DateTime.now().weekday - 1;

  final todayMeal =
      meals.isNotEmpty ? meals[todayIndex] : null;

  final todayWorkout =
      workouts.isNotEmpty ? workouts[todayIndex] : null;

  // -------------------------
  // DIET NOTIFICATION
  // -------------------------

  if (todayMeal != null) {

    final dietMessage =
        'Breakfast: ${todayMeal.breakfast.name}\n'
        'Lunch: ${todayMeal.lunch.name}\n'
        'Dinner: ${todayMeal.dinner.name}';

    await NotificationService.scheduleDietPlan(
      dietMessage,
    );
  }

  // -------------------------
  // WORKOUT NOTIFICATION
  // -------------------------

  if (todayWorkout != null &&
      todayWorkout.exercises.isNotEmpty) {

    final workoutMessage =
        '${todayWorkout.focus}: '
        '${todayWorkout.exercises.map((e) => e.name).join(", ")}';

    await NotificationService.scheduleWorkout(
      workoutMessage,
    );
  }

  _notificationSent[disease] = true;
}
    if (!mounted) return;

setState(() {
  _generatedPlans[disease] =
      _buildPlanWithMeals(
        disease,
        aiMetadata,
        meals,
        workouts,
      );
});
  } catch (e) {

    print("❌ MAIN ERROR");
    print(e);

    _errorMessage = e.toString();

  } finally {

    if (!mounted) return;

    setState(() {
      _isLoading = false;
    });
  }
}
  
  Future<AiHealthPlanData> _generateMetadataFromGroq(
    String disease) async {

  final userData = await SessionService().getUserData();

  final prompt = '''
You are a STRICT JSON API.

Return ONLY valid JSON.

DO NOT:
- explain
- use markdown
- use ```json
- add notes
- add extra text

Generate:
- subtitle
- disease description
- key nutrients
- foods to avoid
- superfoods
- exercise overview
- exercise tips

Disease: $disease
Age: ${userData['age']}
Symptoms: ${userData['symptoms']}
Condition: ${userData['condition']}
Goal: ${userData['goal'] ?? 'general wellness'}

Return ONLY this JSON structure:

{
  "subtitle":"PCOS Wellness Plan",
  "description":"Short disease description",
  "keyNutrients":["item1","item2"],
  "foodsToAvoid":["item1","item2"],
  "superfoods":["item1","item2"],
  "exerciseOverview":"Short overview",
  "exerciseTips":["tip1","tip2"]
}
''';

  final response =
      await GroqService().sendHealthPlanMessage(prompt);

  print("METADATA RAW RESPONSE:");
  print(response);

  final cleaned = _extractJson(response);

  print("METADATA CLEANED:");
  print(cleaned);

  Map<String, dynamic> data = {};

  try {
    final decoded = jsonDecode(cleaned);

if (decoded is! Map<String, dynamic>) {
  throw Exception("Metadata is not valid JSON object");
}

data = decoded;
  } catch (e) {

    print("❌ METADATA JSON ERROR");
    print(e);

    return const AiHealthPlanData(
      subtitle: 'Health Plan',
      description: 'Healthy lifestyle guidance',

      keyNutrients: [],
      foodsToAvoid: [],
      superfoods: [],

      exerciseOverview:
          'Regular exercise recommended.',

      exerciseTips: [],
    );
  }

  // SAFETY CHECK
  if (data == null || data is! Map) {

    print("❌ Metadata is NULL or INVALID");

    return const AiHealthPlanData(
      subtitle: 'Health Plan',
      description: 'Healthy lifestyle guidance',

      keyNutrients: [],
      foodsToAvoid: [],
      superfoods: [],

      exerciseOverview:
          'Regular exercise recommended.',

      exerciseTips: [],
    );
  }

  return AiHealthPlanData(
    subtitle:
        (data['subtitle'] ?? 'Health Plan')
            .toString(),

    description:
        (data['description'] ??
                'Healthy lifestyle guidance')
            .toString(),

    keyNutrients:
        List<String>.from(
            data['keyNutrients'] ?? []),

    foodsToAvoid:
        List<String>.from(
            data['foodsToAvoid'] ?? []),

    superfoods:
        List<String>.from(
            data['superfoods'] ?? []),

    exerciseOverview:
        (data['exerciseOverview'] ??
                'Regular exercise recommended.')
            .toString(),

    exerciseTips:
        List<String>.from(
            data['exerciseTips'] ?? []),
  );
}

  Future<void> fetchAllDietPlans() async {

  setState(() {
    _isLoading = true;
    _errorMessage = '';
  });

  try {

    for (String disease in _diseases) {

      // AI METADATA
      AiHealthPlanData aiMetadata;

try {

  aiMetadata =
      await _generateMetadataFromGroq(
          disease);

  print(
    '✓ Metadata generated for $disease');

} catch (e) {

  print(
    '⚠ Metadata generation failed for $disease');

  print(e);

  aiMetadata = const AiHealthPlanData(
    subtitle: 'Personalized Health Plan',

    description:
        'AI-generated wellness guidance.',

    keyNutrients: [],

    foodsToAvoid: [],

    superfoods: [],

    exerciseOverview:
        'Light daily movement recommended.',

    exerciseTips: [],
  );
}

      // AI MEALS
      List<DayPlan> weeklyMeals = [];

      try {
        weeklyMeals = await _generateWeeklyMealsSafe(disease);
        print('✓ AI meals generated for $disease');
      } catch (e) {
        print('⚠ Meal generation failed for $disease');
        print(e);

        weeklyMeals = [];
      }

      // AI WORKOUTS
      List<WorkoutDay> weeklyWorkouts = [];

      try {

        weeklyWorkouts =
            await _generateWorkoutFromGroq(
                disease);

        print(
          '✓ AI workouts generated for $disease');

      } catch (e) {

        print(
          '⚠ Workout generation failed for $disease');

        print(e);

        weeklyWorkouts = [
  WorkoutDay(
    day: 'Monday',
    focus: 'Light Exercise',
    exercises: [
      Exercise(
        name: 'Walking',
        duration: '20 mins',
        intensity: 'Low',
        benefit: 'Improves circulation',
        icon: Icons.directions_walk,
        steps: [
          'Warm up',
          'Walk slowly',
          'Cool down',
        ],
      ),
    ],
  ),
];
      }

      // BUILD FINAL PLAN
      _generatedPlans[disease] =
          _buildPlanWithMeals(
        disease,
        aiMetadata,
        weeklyMeals,
        weeklyWorkouts,
      );

      if (disease != _diseases.last) {
        await Future.delayed(const Duration(seconds: 2));
      }
    }

    setState(() {
      _isLoading = false;
    });

  } catch (e) {

    setState(() {

      _errorMessage =
          'Failed to load plans: $e';

      _isLoading = false;
    });
  }
}


Future<List<DayPlan>> _generateWeeklyMealsSafe(String disease) async {
  const days = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
  ];

  final userData = await SessionService().getUserData();

  final prompt = '''
You are a clinical nutrition expert.

Return ONLY valid JSON.

Generate a medically accurate 7-day meal plan specifically for:

DISEASE: $disease

IMPORTANT RULES:
- Meals MUST be different based on disease
- Include foods scientifically beneficial for $disease
- Avoid foods harmful for $disease
- Meals must NOT be generic
- Every disease should produce unique meals
- Keep meals realistic and healthy
- Keep descriptions short
- Use 2 to 3 food items only
- No markdown
- No explanation
- No extra text

USER DETAILS:
Age: ${userData['age']}
Symptoms: ${userData['symptoms']}
Condition: ${userData['condition']}
Goal: ${userData['goal'] ?? 'general wellness'}

DISEASE-SPECIFIC REQUIREMENTS:

For PCOS:
- low glycemic foods
- high fiber
- omega 3 rich foods
- avoid sugar and refined carbs

For Endometriosis:
- anti inflammatory foods
- omega 3 foods
- avoid red meat and processed foods

For Cervical Cancer:
- antioxidant rich foods
- vitamin C and folate foods
- immune boosting foods

Return EXACTLY 7 items from Monday to Sunday.

Return ONLY this JSON array format:

[
  {
    "day":"Monday",
    "breakfast":{
      "name":"...",
      "description":"...",
      "calories":"...",
      "items":["...","..."]
    },
    "lunch":{
      "name":"...",
      "description":"...",
      "calories":"...",
      "items":["...","..."]
    },
    "dinner":{
      "name":"...",
      "description":"...",
      "calories":"...",
      "items":["...","..."]
    },
    "snack":{
      "name":"...",
      "description":"...",
      "calories":"...",
      "items":["...","..."]
    }
  }
]
''';

  String response = '';

  for (int retry = 0; retry < 3; retry++) {
    try {
      response = await GroqService().sendHealthPlanMessage(
        prompt,
        maxTokens: 2500,
      );
      final cleaned = _extractJson(response);
      final decoded = jsonDecode(cleaned);

        final List<dynamic> weekData = decoded is Map && decoded['week'] is List
          ? decoded['week'] as List
          : decoded is List
            ? decoded
            : decoded is Map && decoded.containsKey('day')
              ? [decoded]
              : [];

      Meal parseMeal(dynamic m) {
        if (m is! Map) {
          return const Meal(name: 'Unavailable', description: '', calories: '0', items: []);
        }
        return Meal(
          name: (m['name'] ?? '').toString(),
          description: (m['description'] ?? '').toString(),
          calories: (m['calories'] ?? '0').toString(),
          items: (m['items'] is List)
              ? List<String>.from(m['items'].map((e) => e.toString()))
              : [],
        );
      }

      final Map<String, Map<String, dynamic>> mealsByDay = {};
      for (final item in weekData) {
        if (item is Map && item['day'] != null) {
          mealsByDay[item['day'].toString()] = Map<String, dynamic>.from(item);
        }
      }

      if (mealsByDay.isEmpty) {
        throw Exception('Weekly meal JSON did not contain any day entries');
      }

      return days.map((day) {
        final data = mealsByDay[day];
        if (data == null) {
          return DayPlan(
            day: day,
            breakfast: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
            lunch: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
            dinner: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
            snack: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
          );
        }

        return DayPlan(
          day: data['day']?.toString() ?? day,
          breakfast: parseMeal(data['breakfast']),
          lunch: parseMeal(data['lunch']),
          dinner: parseMeal(data['dinner']),
          snack: parseMeal(data['snack']),
        );
      }).toList();
    } catch (e) {
      print('❌ Weekly meal retry ${retry + 1} failed');
      print(e);
    }
  }

  return days
      .map(
        (day) => DayPlan(
          day: day,
          breakfast: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
          lunch: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
          dinner: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
          snack: const Meal(name: 'Unavailable', description: '', calories: '0', items: []),
        ),
      )
      .toList();
}

 Future<List<WorkoutDay>> _generateWorkoutFromGroq(
    String disease) async {

  final userData =
      await SessionService().getUserData();

  final prompt = '''
You are a STRICT JSON API.

Return ONLY valid JSON array.

Generate SHORT compact 7-day workout JSON.

Maximum:
- 2 exercises per day
- 3 steps only
- 5 words per field

For rest day use:
"exercises":[]
- Keep JSON compact
- No markdown
- No explanation
- No extra text

Disease: $disease
Age: ${userData['age']}
Symptoms: ${userData['symptoms']}
Condition: ${userData['condition']}
Fitness Level: ${userData['fitnessLevel']}

Return ONLY:

[
  {
    "day":"Monday",
    "focus":"Strength",
    "exercises":[
      {
        "name":"Walking",
        "duration":"20 minutes",
        "intensity":"Low",
        "benefit":"Improves circulation",
        "steps":[
          "Warm up",
          "Walk",
          "Cool down"
        ]
      }
    ]
  }
]
''';

  String response = '';

  dynamic decodedRaw;

  bool success = false;

  for (int retry = 0; retry < 3; retry++) {

    try {

      response =
          await GroqService()
              .sendHealthPlanMessage(prompt);

      print("WORKOUT RAW RESPONSE:");
      print(response);

      final cleaned =
          _extractJson(response);

      print("WORKOUT CLEANED JSON:");
      print(cleaned);

try {

  decodedRaw = jsonDecode(cleaned);
  if (decodedRaw == null) {
  throw Exception("Workout JSON is null");
}

if (decodedRaw is! List &&
    decodedRaw is! Map) {
  throw Exception("Workout JSON invalid");
}

} catch (e) {

  print("❌ WORKOUT JSON ERROR");
  print(e);
  print(cleaned);

  return [];
}

      success = true;
      break;

    } catch (e) {

      print(
          "❌ Workout retry ${retry + 1}");

      print(e);
    }
  }

  if (!success) {

    print(
        "❌ Workout generation completely failed");

    return [
  WorkoutDay(
    day: 'Monday',
    focus: 'Light Exercise',
    exercises: [
      Exercise(
        name: 'Walking',
        duration: '20 mins',
        intensity: 'Low',
        benefit: 'Improves circulation',
        icon: Icons.directions_walk,
        steps: [
          'Warm up',
          'Walk',
          'Cool down',
        ],
      ),
    ],
  ),
];
  }
  
  final List<dynamic> decoded =
      decodedRaw is List
          ? decodedRaw
          : [decodedRaw];

  List<WorkoutDay> workoutDays = [];

  for (var day in decoded) {

    if (day['day'] == null) continue;

if (day['focus'] == null) {
  day['focus'] = 'General Fitness';
}

if (day['exercises'] == null) {
  day['exercises'] = [];
}

    if (day is! Map) continue;

    List<Exercise> exercises = [];

    final exList =
        day['exercises'];

    if (exList is List) {

      for (var ex in exList) {

        if (ex['name'] == null) continue;
        if (ex is! Map) continue;

        exercises.add(
          Exercise(
            name:
                (ex['name'] ?? '')
                    .toString(),

            duration:
                (ex['duration'] ?? '')
                    .toString(),

            intensity:
                (ex['intensity'] ?? '')
                    .toString(),

            benefit:
                (ex['benefit'] ?? '')
                    .toString(),

            icon:
                _getExerciseIcon(
                    ex['name'] ?? ''),

            steps:
                (ex['steps'] is List)
                    ? List<String>.from(
                        ex['steps']
                            .map((e) =>
                                e.toString()))
                    : [],
          ),
        );
      }
    }

    workoutDays.add(
      WorkoutDay(
        day:
            (day['day'] ?? '')
                .toString(),

        focus:
            (day['focus'] ?? '')
                .toString(),

        exercises: exercises,
      ),
    );
  }
  const allDays = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];

for (String d in allDays) {

  final exists =
      workoutDays.any((w) => w.day == d);

  if (!exists) {

    workoutDays.add(
      WorkoutDay(
        day: d,
        focus: 'Recovery',
        exercises: [],
      ),
    );
  }
}

workoutDays.sort(
  (a, b) =>
      allDays.indexOf(a.day)
          .compareTo(
              allDays.indexOf(b.day)),
);
  return workoutDays;
}

String _extractJson(String response) {

  if (response.trim().isEmpty) {
    throw Exception("Empty AI response");
  }

  // Remove markdown
  String cleaned = response
      .replaceAll(RegExp(r'```json'), '')
      .replaceAll(RegExp(r'```'), '')
      .trim();

  // Find first JSON object OR array
  final objectStart = cleaned.indexOf('{');
  final arrayStart = cleaned.indexOf('[');

  int start = -1;

  if (objectStart == -1 && arrayStart == -1) {
    throw Exception("No JSON found");
  }

  if (objectStart == -1) {
    start = arrayStart;
  } else if (arrayStart == -1) {
    start = objectStart;
  } else {
    start = objectStart < arrayStart
        ? objectStart
        : arrayStart;
  }

  final opening = cleaned[start];
  final closing = opening == '{' ? '}' : ']';

  int balance = 0;
  int end = -1;

  for (int i = start; i < cleaned.length; i++) {

    final char = cleaned[i];

    if (char == opening) {
      balance++;
    }

    if (char == closing) {
      balance--;

      if (balance == 0) {
        end = i;
        break;
      }
    }
  }

  if (end == -1) {
    throw Exception("Incomplete JSON");
  }

  String jsonString =
      cleaned.substring(start, end + 1);

  // Remove trailing commas
  jsonString = jsonString
      .replaceAll(RegExp(r',\s*}'), '}')
      .replaceAll(RegExp(r',\s*]'), ']');

  return jsonString.trim();
}

DiseaseDietPlan _buildPlanWithMeals(
  String disease,
  AiHealthPlanData aiData,
  List<DayPlan> meals,
  List<WorkoutDay> workouts,
) {

  final uiData =
      _diseaseMetadata[disease] ??
      _diseaseMetadata['PCOS']!;

  return DiseaseDietPlan(
    disease: disease,

    // AI GENERATED
    subtitle: aiData.subtitle,
    description: aiData.description,

    keyNutrients:
        aiData.keyNutrients,

    foodsToAvoid:
        aiData.foodsToAvoid,

    superfoods:
        aiData.superfoods,

    exerciseOverview:
        aiData.exerciseOverview,

    exerciseTips:
        aiData.exerciseTips,

    // UI ONLY
    icon: uiData['icon'],
    primaryColor:
        uiData['primaryColor'],
    accentColor:
        uiData['accentColor'],

    // AI GENERATED
    weeklyPlan: meals,
    workoutWeek: workouts,
  );
}

  Meal _parseMeal(dynamic mealData) {

  if (mealData == null) {

    return const Meal(
      name: 'Meal',
      description: 'Coming soon',
      calories: '0',
      items: [],
    );
  }

  return Meal(
    name: mealData['name'] ?? 'Meal',

    description:
        mealData['description'] ?? 'Description',

    calories:
        mealData['calories']
            ?.toString() ?? '0',

    items:
        List<String>.from(
            mealData['items'] ?? []),
  );
}

  IconData _getExerciseIcon(String name) {
    if (name.toLowerCase().contains('walk')) return Icons.directions_walk;
    if (name.toLowerCase().contains('swim')) return Icons.pool;
    if (name.toLowerCase().contains('yoga') || name.toLowerCase().contains('pose')) return Icons.self_improvement;
    if (name.toLowerCase().contains('run')) return Icons.directions_run;
    if (name.toLowerCase().contains('cycle') || name.toLowerCase().contains('bike')) return Icons.pedal_bike;
    return Icons.fitness_center;
  }

  DiseaseDietPlan get _currentPlan {
    return _generatedPlans[_currentDisease] ?? _getDefaultPlan(_currentDisease);
  }

  DiseaseDietPlan _getDefaultPlan(
    String disease) {

  final metadata =
      _diseaseMetadata[disease] ??
      _diseaseMetadata['PCOS']!;

  return DiseaseDietPlan(
    disease: disease,

    subtitle:
        'Personalized Health Plan',

    description:
        'AI-generated wellness guidance.',

    icon: metadata['icon'],

    primaryColor:
        metadata['primaryColor'],

    accentColor:
        metadata['accentColor'],

    keyNutrients: [],

    foodsToAvoid: [],

    superfoods: [],

    weeklyPlan: [],

    exerciseOverview:
        'Light daily exercise recommended.',

    exerciseTips: [],

    workoutWeek: [],
  );
}

  Color _intensityColor(String intensity) {
    switch (intensity) {
      case 'High': return const Color(0xFFE53E3E);
      case 'Moderate': return const Color(0xFFD97706);
      default: return const Color(0xFF2D9E6B);
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final diet = Provider.of<AppState>(context).dietPlan;
    final workout = Provider.of<AppState>(context).workoutPlan;
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      body: SafeArea(
        child: _isLoading
            ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 16),
                    const Text('Generating your personalized plan...'),
                  ],
                ),
              )
            : _errorMessage.isNotEmpty
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.error_outline, color: Colors.red, size: 48),
                          const SizedBox(height: 16),
                          const Text('Could not generate the plan', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 8),
                          Text(_errorMessage, textAlign: TextAlign.center, style: const TextStyle(color: Colors.grey)),
                          const SizedBox(height: 20),
                          ElevatedButton.icon(
                            onPressed: fetchAllDietPlans,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Try again'),
                          ),
                        ],
                      ),
                    ),
                  )
                : Column(
                    children: [
                      _buildHeader(),
                      _buildDiseaseTabBar(),
                      Expanded(
                        child: _currentPlan == null || _currentPlan.weeklyPlan.isEmpty
                            ? const Center(child: Text('No plan generated'))
                            : SingleChildScrollView(
                                child: Column(
                                  children: [
                                    _buildDietOverviewCard(),
                                    _buildSuperFoodsSection(),
                                    _buildFoodsToAvoidSection(),
                                    _buildWeeklyPlanSection(),
                                    _buildNutrientsSection(),
                                    _buildExerciseSection(),
                                    const SizedBox(height: 24),
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
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [_currentPlan.primaryColor, _currentPlan.primaryColor.withOpacity(0.7)],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.arrow_back, color: Colors.white, size: 26),
          ),
          const SizedBox(width: 8),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Diet & Exercise Guide', style: TextStyle(color: Colors.white, fontSize: 21, fontWeight: FontWeight.w800)),
              Text('Personalized wellness for women\'s health', style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 12)),
            ],
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
            child: Icon(_currentPlan.icon, color: Colors.white, size: 26),
          ),
        ],
      ),
    );
  }

  Widget _buildDiseaseTabBar() {
    return Container(
      color: Colors.white,
      child: TabBar(
        controller: _tabController,
        isScrollable: true,
        indicatorColor: _currentPlan.primaryColor,
        indicatorWeight: 3,
        labelColor: _currentPlan.primaryColor,
        unselectedLabelColor: Colors.grey,
        labelStyle: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13),
        unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w500, fontSize: 12),
        tabs: _diseases.map((disease) => Tab(child: Padding(padding: const EdgeInsets.symmetric(horizontal: 4), child: Text(disease)))).toList(),
      ),
    );
  }

  Widget _buildDietOverviewCard() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [_currentPlan.primaryColor, _currentPlan.primaryColor.withOpacity(0.75)]),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: _currentPlan.primaryColor.withOpacity(0.3), blurRadius: 15, offset: const Offset(0, 5))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(color: Colors.white.withOpacity(0.25), borderRadius: BorderRadius.circular(12)),
                child: Icon(_currentPlan.icon, color: Colors.white, size: 28),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_currentPlan.disease, style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
                    Text(_currentPlan.subtitle, style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 12)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(12)),
            child: Text(_currentPlan.description, style: const TextStyle(color: Colors.white, fontSize: 13.5, height: 1.5)),
          ),
        ],
      ),
    );
  }

  Widget _buildSuperFoodsSection() {
    return _buildSection(
      title: '⭐ Superfoods to Include',
      color: _currentPlan.primaryColor,
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: _currentPlan.superfoods.map((food) => Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: _currentPlan.accentColor,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: _currentPlan.primaryColor.withOpacity(0.3)),
          ),
          child: Text(food, style: TextStyle(color: _currentPlan.primaryColor, fontWeight: FontWeight.w600, fontSize: 13)),
        )).toList(),
      ),
    );
  }

  Widget _buildFoodsToAvoidSection() {
    return _buildSection(
      title: '🚫 Foods to Avoid',
      color: _currentPlan.primaryColor,
      child: Column(
        children: _currentPlan.foodsToAvoid.map((food) => Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(
            children: [
              Container(width: 8, height: 8, decoration: const BoxDecoration(color: Color(0xFFE53E3E), shape: BoxShape.circle)),
              const SizedBox(width: 12),
              Expanded(child: Text(food, style: const TextStyle(fontSize: 14, color: Color(0xFF444444)))),
            ],
          ),
        )).toList(),
      ),
    );
  }

Widget _buildWeeklyPlanSection() {

  // SAFETY CHECK
  if (_currentPlan.weeklyPlan.isEmpty) {

    return _buildSection(
      title: '📅 Weekly Meal Plan',
      color: _currentPlan.primaryColor,
      child: const Center(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Text(
            'No meal plan available',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }

  // SAFE INDEX
  final safeIndex =
      _selectedDayIndex >= _currentPlan.weeklyPlan.length
          ? 0
          : _selectedDayIndex;

  final selectedDay =
      _currentPlan.weeklyPlan[safeIndex];

  return _buildSection(
    title: '📅 Weekly Meal Plan',
    color: _currentPlan.primaryColor,
    child: Column(
      children: [

        SizedBox(
          height: 42,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: _currentPlan.weeklyPlan.length,
            itemBuilder: (context, index) {

              final isSelected =
                  index == safeIndex;

              return GestureDetector(

                onTap: () {

                  setState(() {
                    _selectedDayIndex = index;
                  });
                },

                child: AnimatedContainer(

                  duration:
                      const Duration(milliseconds: 200),

                  margin:
                      const EdgeInsets.only(right: 8),

                  padding:
                      const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),

                  decoration: BoxDecoration(
                    color: isSelected
                        ? _currentPlan.primaryColor
                        : Colors.white,

                    borderRadius:
                        BorderRadius.circular(20),

                    border: Border.all(
                      color: _currentPlan.primaryColor,
                      width: 1.5,
                    ),
                  ),

                  child: Text(

                    _currentPlan
                        .weeklyPlan[index]
                        .day,

                    style: TextStyle(
                      color: isSelected
                          ? Colors.white
                          : _currentPlan.primaryColor,

                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                ),
              );
            },
          ),
        ),

        const SizedBox(height: 16),

        _buildMealCard(
          '🌅 Breakfast',
          selectedDay.breakfast,
          const Color(0xFFFFF3CD),
          const Color(0xFFD97706),
        ),

        const SizedBox(height: 12),

        _buildMealCard(
          '☀️ Lunch',
          selectedDay.lunch,
          const Color(0xFFD1FAE5),
          const Color(0xFF059669),
        ),

        const SizedBox(height: 12),

        _buildMealCard(
          '🌙 Dinner',
          selectedDay.dinner,
          const Color(0xFFEDE9FE),
          const Color(0xFF7C3AED),
        ),

        const SizedBox(height: 12),

        _buildMealCard(
          '🍎 Snack',
          selectedDay.snack,
          const Color(0xFFFFE4E6),
          const Color(0xFFE11D48),
        ),
      ],
    ),
  );
}
  Widget _buildMealCard(String label, Meal meal, Color bgColor, Color accentColor) {
    return Container(
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: accentColor.withOpacity(0.3)),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
          leading: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: accentColor.withOpacity(0.15), borderRadius: BorderRadius.circular(10)),
            child: Text(label.split(' ')[0], style: const TextStyle(fontSize: 20)),
          ),
          title: Text(meal.name, maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15, color: accentColor)),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(meal.description, maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: accentColor.withOpacity(0.8))),
                const SizedBox(height: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(color: accentColor.withOpacity(0.15), borderRadius: BorderRadius.circular(10)),
                  child: Text(meal.calories, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: accentColor)),
                ),
              ],
            ),
          ),
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Ingredients & Portions:', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13, color: accentColor)),
                const SizedBox(height: 8),
                ...meal.items.map((item) => Padding(
                  padding: const EdgeInsets.only(bottom: 5),
                  child: Row(
                    children: [
                      Icon(Icons.check_circle_outline, size: 16, color: accentColor),
                      const SizedBox(width: 8),
                      Expanded(child: Text(item, style: TextStyle(fontSize: 13, color: accentColor.withOpacity(0.9)))),
                    ],
                  ),
                )),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNutrientsSection() {
    return _buildSection(
      title: '💊 Key Nutrients',
      color: _currentPlan.primaryColor,
      child: GridView.count(
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        crossAxisCount: 2,
        mainAxisSpacing: 10,
        crossAxisSpacing: 10,
        childAspectRatio: 3,
        children: _currentPlan.keyNutrients.map((nutrient) => Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(color: _currentPlan.accentColor, borderRadius: BorderRadius.circular(12), border: Border.all(color: _currentPlan.primaryColor.withOpacity(0.2))),
          child: Row(
            children: [
              Icon(Icons.local_pharmacy_outlined, size: 16, color: _currentPlan.primaryColor),
              const SizedBox(width: 6),
              Expanded(child: Text(nutrient, style: TextStyle(color: _currentPlan.primaryColor, fontWeight: FontWeight.w600, fontSize: 12), overflow: TextOverflow.ellipsis)),
            ],
          ),
        )).toList(),
      ),
    );
  }

  Widget _buildExerciseSection() {

  final plan = _currentPlan;

  // SAFETY CHECK
  if (plan.workoutWeek.isEmpty) {

    return _buildSection(
      title: '🏋️ Exercise & Workout Plan',
      color: plan.primaryColor,
      child: const Center(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Text(
            'No workout plan available',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }

  // SAFE INDEX
  final safeWorkoutIndex =
      _selectedWorkoutDayIndex >= plan.workoutWeek.length
          ? 0
          : _selectedWorkoutDayIndex;

  final workout =
      plan.workoutWeek[safeWorkoutIndex];

  return _buildSection(
    title: '🏋️ Exercise & Workout Plan',
    color: plan.primaryColor,

    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,

      children: [

        // OVERVIEW
        Container(

          padding: const EdgeInsets.all(14),

          decoration: BoxDecoration(
            color: plan.accentColor,

            borderRadius:
                BorderRadius.circular(14),

            border: Border.all(
              color:
                  plan.primaryColor.withOpacity(0.25),
            ),
          ),

          child: Row(
            crossAxisAlignment:
                CrossAxisAlignment.start,

            children: [

              Icon(
                Icons.info_outline_rounded,
                color: plan.primaryColor,
                size: 18,
              ),

              const SizedBox(width: 10),

              Expanded(
                child: Text(

                  plan.exerciseOverview,

                  style: TextStyle(
                    fontSize: 13,
                    color:
                        plan.primaryColor.withOpacity(
                            0.85),
                    height: 1.5,
                  ),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 16),

        // TIPS TITLE
        Text(
          '💡 Exercise Tips',

          style: TextStyle(
            fontWeight: FontWeight.w800,
            fontSize: 14,
            color: plan.primaryColor,
          ),
        ),

        const SizedBox(height: 8),

        // TIPS
        ...plan.exerciseTips.map(

          (tip) => Padding(

            padding:
                const EdgeInsets.only(bottom: 7),

            child: Row(
              crossAxisAlignment:
                  CrossAxisAlignment.start,

              children: [

                Container(

                  margin:
                      const EdgeInsets.only(top: 5),

                  width: 7,
                  height: 7,

                  decoration: BoxDecoration(
                    color: plan.primaryColor,
                    shape: BoxShape.circle,
                  ),
                ),

                const SizedBox(width: 10),

                Expanded(
                  child: Text(

                    tip,

                    style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFF444444),
                      height: 1.4,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),

        // WORKOUT SCHEDULE
        const SizedBox(height: 20),

        Text(
          '📆 Weekly Workout Schedule',

          style: TextStyle(
            fontWeight: FontWeight.w800,
            fontSize: 14,
            color: plan.primaryColor,
          ),
        ),

        const SizedBox(height: 12),

        SizedBox(

          height: 42,

          child: ListView.builder(

            scrollDirection: Axis.horizontal,

            itemCount: plan.workoutWeek.length,

            itemBuilder: (context, index) {

              final isSelected =
                  index == safeWorkoutIndex;

              return GestureDetector(

                onTap: () {

                  setState(() {
                    _selectedWorkoutDayIndex =
                        index;
                  });
                },

                child: AnimatedContainer(

                  duration:
                      const Duration(milliseconds: 200),

                  margin:
                      const EdgeInsets.only(right: 8),

                  padding:
                      const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),

                  decoration: BoxDecoration(

                    color: isSelected
                        ? plan.primaryColor
                        : Colors.white,

                    borderRadius:
                        BorderRadius.circular(20),

                    border: Border.all(
                      color: plan.primaryColor,
                      width: 1.5,
                    ),
                  ),

                  child: Text(

                    plan
                        .workoutWeek[index]
                        .day,

                    style: TextStyle(
                      color: isSelected
                          ? Colors.white
                          : plan.primaryColor,

                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                ),
              );
            },
          ),
        ),

        const SizedBox(height: 12),

        // FOCUS CHIP
        Container(

          padding: const EdgeInsets.symmetric(
            horizontal: 14,
            vertical: 8,
          ),

          decoration: BoxDecoration(

            gradient: LinearGradient(
              colors: [
                plan.primaryColor,
                plan.primaryColor.withOpacity(0.7),
              ],
            ),

            borderRadius:
                BorderRadius.circular(20),
          ),

          child: Row(
            mainAxisSize: MainAxisSize.min,

            children: [

              const Icon(
                Icons.flash_on_rounded,
                color: Colors.white,
                size: 16,
              ),

              const SizedBox(width: 6),

              Text(

                'Focus: ${workout.focus}',

                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 14),

        // EXERCISES
        if (workout.exercises.isEmpty)

          const Padding(

            padding: EdgeInsets.all(16),

            child: Text(
              'Rest day / No exercises',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          )

        else

          ...workout.exercises.map(

            (exercise) => _buildExerciseCard(
              exercise,
              plan.primaryColor,
              plan.accentColor,
            ),
          ),
      ],
    ),
  );
}

  Widget _buildExerciseCard(Exercise exercise, Color primary, Color accent) {
    final intensityColor = _intensityColor(exercise.intensity);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: accent,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: primary.withOpacity(0.2), width: 1.5),
        boxShadow: [BoxShadow(color: primary.withOpacity(0.06), blurRadius: 8, offset: const Offset(0, 3))],
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.fromLTRB(14, 6, 14, 6),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
          leading: Container(
            padding: const EdgeInsets.all(9),
            decoration: BoxDecoration(color: primary.withOpacity(0.12), borderRadius: BorderRadius.circular(11)),
            child: Icon(exercise.icon, color: primary, size: 20),
          ),
          title: Text(exercise.name, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: primary)),
          subtitle: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 4),
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(color: primary.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.timer_outlined, size: 11, color: primary),
                        const SizedBox(width: 4),
                        Text(exercise.duration, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: primary)),
                      ],
                    ),
                  ),
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(color: intensityColor.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.bolt_rounded, size: 11, color: intensityColor),
                        const SizedBox(width: 3),
                        Text(exercise.intensity, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: intensityColor)),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 5),
              Text(exercise.benefit, style: TextStyle(fontSize: 11.5, color: primary.withOpacity(0.75), height: 1.3)),
            ],
          ),
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: primary.withOpacity(0.15))),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('How to do it:', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 12, color: primary)),
                  const SizedBox(height: 8),
                  ...List.generate(exercise.steps.length, (i) => Padding(
                    padding: const EdgeInsets.only(bottom: 7),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          width: 20,
                          height: 20,
                          decoration: BoxDecoration(color: primary, shape: BoxShape.circle),
                          child: Center(child: Text('${i + 1}', style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold))),
                        ),
                        const SizedBox(width: 10),
                        Expanded(child: Text(exercise.steps[i], style: const TextStyle(fontSize: 13, color: Color(0xFF333333), height: 1.4))),
                      ],
                    ),
                  )),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSection({required String title, required Color color, required Widget child}) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 4, height: 20, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
              const SizedBox(width: 10),
              Text(title, style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: color)),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}