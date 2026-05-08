import 'package:flutter/material.dart';
import 'services/groq_service.dart';
import 'services/session_service.dart';
import 'dart:convert';

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
  const DayPlan({required this.day, required this.breakfast, required this.lunch, required this.dinner, required this.snack});
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

  'Thyroid': {
    'icon': Icons.self_improvement,
    'primaryColor': const Color(0xFF2E86AB),
    'accentColor': const Color(0xFFE0F4FF),
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
  
  final List<String> _diseases = ['PCOS', 'Endometriosis', 'Thyroid', 'Cervical Cancer'];
  String _currentDisease = 'PCOS';

  @override

  void initState() {
  super.initState();

  _tabController = TabController(
    length: _diseases.length,
    vsync: this,
  );

  _tabController.addListener(_onDiseaseChanged);

  // Generate ONLY current tab
  _generateCurrentDiseasePlan();
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

    // METADATA
    AiHealthPlanData aiMetadata;

    try {

      aiMetadata =
          await _generateMetadataFromGroq(
              disease);

    } catch (_) {

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

    // MEALS
    List<DayPlan> meals = [];

    try {

      meals =
          await _generateMealsFromGroq(
              disease);

    } catch (e) {

      print(e);
    }

    // WORKOUTS
    List<WorkoutDay> workouts = [];

    try {

      workouts =
          await _generateWorkoutFromGroq(
              disease);

    } catch (e) {

      print(e);
    }

    _generatedPlans[disease] =
        _buildPlanWithMeals(
      disease,
      aiMetadata,
      meals,
      workouts,
    );

  } catch (e) {

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
You are a JSON API.

Return ONLY valid JSON.

Do not explain.
Do not use markdown.
Do not write ```json.
Do not add extra text.

Disease: $disease
Age: ${userData['age']}
Symptoms: ${userData['symptoms']}
Condition: ${userData['condition']}
Goal: ${userData['goal'] ?? 'general wellness'}

Generate:
- subtitle
- disease description
- key nutrients
- foods to avoid
- superfoods
- exercise overview
- exercise tips

Return JSON format:

{
  "subtitle":"",
  "description":"",
  "keyNutrients":[""],
  "foodsToAvoid":[""],
  "superfoods":[""],
  "exerciseOverview":"",
  "exerciseTips":[""]
}
''';

  final response =
      await GroqService().sendHealthPlanMessage(prompt);

  final cleaned = _extractJson(response);

  final data = jsonDecode(cleaned);

  return AiHealthPlanData(
    subtitle: data['subtitle'] ?? '',
    description: data['description'] ?? '',
    keyNutrients:
        List<String>.from(data['keyNutrients'] ?? []),
    foodsToAvoid:
        List<String>.from(data['foodsToAvoid'] ?? []),
    superfoods:
        List<String>.from(data['superfoods'] ?? []),
    exerciseOverview:
        data['exerciseOverview'] ?? '',
    exerciseTips:
        List<String>.from(data['exerciseTips'] ?? []),
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

        weeklyMeals =
            await _generateMealsFromGroq(
                disease);

        print(
          '✓ AI meals generated for $disease');

      } catch (e) {

        print(
          '⚠ Meal generation failed for $disease');

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

        weeklyWorkouts = [];
      }

      // BUILD FINAL PLAN
      _generatedPlans[disease] =
          _buildPlanWithMeals(
        disease,
        aiMetadata,
        weeklyMeals,
        weeklyWorkouts,
      );
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

 Future<List<DayPlan>> _generateMealsFromGroq(
    String disease) async {

  final userData =
      await SessionService().getUserData();

  final prompt = '''
You are a JSON API.

Return ONLY valid JSON.

Generate a 3-day meal plan.

Disease: $disease
Age: ${userData['age']}
Condition: ${userData['condition']}
Symptoms: ${userData['symptoms']}

Return EXACTLY this structure:

[
  {
    "day":"Day 1",
    "breakfast":{
      "name":"",
      "description":"",
      "calories":"",
      "items":[]
    },
    "lunch":{
      "name":"",
      "description":"",
      "calories":"",
      "items":[]
    },
    "dinner":{
      "name":"",
      "description":"",
      "calories":"",
      "items":[]
    },
    "snack":{
      "name":"",
      "description":"",
      "calories":"",
      "items":[]
    }
  }
]
''';

  final response =
      await GroqService()
          .sendHealthPlanMessage(prompt);

  final cleaned =
      _extractJson(response);

  final dynamic decodedRaw =
    jsonDecode(cleaned);

  final List<dynamic> decoded =
    decodedRaw is List
      ? decodedRaw
      : [decodedRaw];

  return decoded.map((meal) {

    return DayPlan(
      day: meal['day'] ?? '',

      breakfast:
          _parseMeal(meal['breakfast']),

      lunch:
          _parseMeal(meal['lunch']),

      dinner:
          _parseMeal(meal['dinner']),

      snack:
          _parseMeal(meal['snack']),
    );

  }).toList();
}

 Future<List<WorkoutDay>> _generateWorkoutFromGroq(
    String disease) async {

  final userData =
      await SessionService().getUserData();

  final prompt = '''
You are a JSON API.

Return ONLY valid JSON.

Generate a 3-day workout plan.

Disease: $disease
Age: ${userData['age']}
Condition: ${userData['condition']}
Symptoms: ${userData['symptoms']}
Fitness Level: ${userData['fitnessLevel']}

Return EXACTLY this structure:

[
  {
    "day":"Day 1",
    "focus":"Strength",
    "exercises":[
      {
        "name":"Walking",
        "duration":"20 minutes",
        "intensity":"Low",
        "benefit":"Improves circulation",
        "steps":[
          "Warm up",
          "Walk slowly",
          "Cool down"
        ]
      }
    ]
  }
]
''';

  final response =
      await GroqService()
          .sendHealthPlanMessage(prompt);

  final cleaned =
      _extractJson(response);

  final dynamic decodedRaw =
    jsonDecode(cleaned);

  final List<dynamic> decoded =
    decodedRaw is List
      ? decodedRaw
      : [decodedRaw];

  List<WorkoutDay> workoutDays = [];

  for (var day in decoded) {

    List<Exercise> exercises = [];

    for (var ex in (day['exercises'] ?? [])) {

      exercises.add(
        Exercise(
          name: ex['name'] ?? '',
          duration: ex['duration'] ?? '',
          intensity: ex['intensity'] ?? '',
          benefit: ex['benefit'] ?? '',
          icon: _getExerciseIcon(
              ex['name'] ?? ''),
          steps: List<String>.from(
              ex['steps'] ?? []),
        ),
      );
    }

    workoutDays.add(
      WorkoutDay(
        day: day['day'] ?? '',
        focus: day['focus'] ?? '',
        exercises: exercises,
      ),
    );
  }

  return workoutDays;
}

String _extractJson(String response) {

  final cleaned = response
      .replaceAll(RegExp(r'```(?:json)?\s*'), '')
      .replaceAll('```', '')
      .trim();

  if (cleaned.isEmpty) {
    throw Exception('Empty response');
  }

  final startArray = cleaned.indexOf('[');
  final startObject = cleaned.indexOf('{');
  int start = -1;
  String open = '';
  String close = '';

  if (startArray != -1 &&
      (startObject == -1 || startArray < startObject)) {
    start = startArray;
    open = '[';
    close = ']';
  } else if (startObject != -1) {
    start = startObject;
    open = '{';
    close = '}';
  }

  if (start == -1) {
    throw Exception('No JSON start token found');
  }

  int depth = 0;
  int end = -1;
  bool inString = false;
  bool escaped = false;

  for (int i = start; i < cleaned.length; i++) {
    final ch = cleaned[i];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (ch == '\\') {
      escaped = true;
      continue;
    }

    if (ch == '"') {
      inString = !inString;
      continue;
    }

    if (inString) {
      continue;
    }

    if (ch == open) {
      depth++;
    } else if (ch == close) {
      depth--;
      if (depth == 0) {
        end = i;
        break;
      }
    }
  }

  if (end == -1) {
    throw Exception('Unmatched JSON brackets/braces');
  }

  final jsonStr = cleaned.substring(start, end + 1);
  jsonDecode(jsonStr);
  return jsonStr;
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
                        child: _currentPlan.weeklyPlan.isEmpty
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
                final isSelected = index == _selectedDayIndex;
                return GestureDetector(
                  onTap: () => setState(() => _selectedDayIndex = index),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    margin: const EdgeInsets.only(right: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected ? _currentPlan.primaryColor : Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: _currentPlan.primaryColor, width: 1.5),
                    ),
                    child: Text(_currentPlan.weeklyPlan[index].day,
                        style: TextStyle(color: isSelected ? Colors.white : _currentPlan.primaryColor, fontWeight: FontWeight.w700, fontSize: 13)),
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 16),
          _buildMealCard('🌅 Breakfast', _currentPlan.weeklyPlan[_selectedDayIndex].breakfast, const Color(0xFFFFF3CD), const Color(0xFFD97706)),
          const SizedBox(height: 12),
          _buildMealCard('☀️ Lunch', _currentPlan.weeklyPlan[_selectedDayIndex].lunch, const Color(0xFFD1FAE5), const Color(0xFF059669)),
          const SizedBox(height: 12),
          _buildMealCard('🌙 Dinner', _currentPlan.weeklyPlan[_selectedDayIndex].dinner, const Color(0xFFEDE9FE), const Color(0xFF7C3AED)),
          const SizedBox(height: 12),
          _buildMealCard('🍎 Snack', _currentPlan.weeklyPlan[_selectedDayIndex].snack, const Color(0xFFFFE4E6), const Color(0xFFE11D48)),
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
    final workout = plan.workoutWeek.isNotEmpty ? plan.workoutWeek[_selectedWorkoutDayIndex] : null;

    return _buildSection(
      title: '🏋️ Exercise & Workout Plan',
      color: plan.primaryColor,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: plan.accentColor, borderRadius: BorderRadius.circular(14), border: Border.all(color: plan.primaryColor.withOpacity(0.25))),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline_rounded, color: plan.primaryColor, size: 18),
                const SizedBox(width: 10),
                Expanded(child: Text(plan.exerciseOverview, style: TextStyle(fontSize: 13, color: plan.primaryColor.withOpacity(0.85), height: 1.5))),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text('💡 Exercise Tips', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: plan.primaryColor)),
          const SizedBox(height: 8),
          ...plan.exerciseTips.map((tip) => Padding(
            padding: const EdgeInsets.only(bottom: 7),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(margin: const EdgeInsets.only(top: 5), width: 7, height: 7, decoration: BoxDecoration(color: plan.primaryColor, shape: BoxShape.circle)),
                const SizedBox(width: 10),
                Expanded(child: Text(tip, style: const TextStyle(fontSize: 13, color: Color(0xFF444444), height: 1.4))),
              ],
            ),
          )),
          if (plan.workoutWeek.isNotEmpty) ...[
            const SizedBox(height: 20),
            Text('📆 Weekly Workout Schedule', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: plan.primaryColor)),
            const SizedBox(height: 12),
            SizedBox(
              height: 42,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: plan.workoutWeek.length,
                itemBuilder: (context, index) {
                  final isSelected = index == _selectedWorkoutDayIndex;
                  return GestureDetector(
                    onTap: () => setState(() => _selectedWorkoutDayIndex = index),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      margin: const EdgeInsets.only(right: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: isSelected ? plan.primaryColor : Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: plan.primaryColor, width: 1.5),
                      ),
                      child: Text(plan.workoutWeek[index].day, style: TextStyle(color: isSelected ? Colors.white : plan.primaryColor, fontWeight: FontWeight.w700, fontSize: 13)),
                    ),
                  );
                },
              ),
            ),
            if (workout != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(gradient: LinearGradient(colors: [plan.primaryColor, plan.primaryColor.withOpacity(0.7)]), borderRadius: BorderRadius.circular(20)),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.flash_on_rounded, color: Colors.white, size: 16),
                    const SizedBox(width: 6),
                    Text('Focus: ${workout.focus}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 13)),
                  ],
                ),
              ),
              const SizedBox(height: 14),
              ...workout.exercises.map((exercise) => _buildExerciseCard(exercise, plan.primaryColor, plan.accentColor)),
            ],
          ],
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