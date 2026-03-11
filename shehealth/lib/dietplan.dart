import 'package:flutter/material.dart';
import 'services/groq_service.dart';

class Meal {
  final String name;
  final String description;
  final String calories;
  final List<String> items;

  const Meal({
    required this.name,
    required this.description,
    required this.calories,
    required this.items,
  });
}

class DayPlan {
  final String day;
  final Meal breakfast;
  final Meal lunch;
  final Meal dinner;
  final Meal snack;

  const DayPlan({
    required this.day,
    required this.breakfast,
    required this.lunch,
    required this.dinner,
    required this.snack,
  });
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

  const DiseaseDietPlan({
    required this.disease,
    required this.subtitle,
    required this.description,
    required this.icon,
    required this.primaryColor,
    required this.accentColor,
    required this.keyNutrients,
    required this.foodsToAvoid,
    required this.superfoods,
    required this.weeklyPlan,
  });
}

// ─────────────────────────────────────────────
//  DIET DATA
// ─────────────────────────────────────────────

final List<DiseaseDietPlan> dietPlans = [
  DiseaseDietPlan(
    disease: 'PCOS / PCOD',
    subtitle: 'Polycystic Ovarian Syndrome',
    description:
        'A low-GI, anti-inflammatory diet helps regulate insulin levels and balance hormones naturally.',
    icon: Icons.spa_outlined,
    primaryColor: const Color(0xFFC85A7A),
    accentColor: const Color(0xFFFFE4EC),
    keyNutrients: [
      'Omega-3 Fatty Acids',
      'Magnesium',
      'Zinc',
      'Inositol (B8)',
      'Chromium',
      'Vitamin D',
    ],
    foodsToAvoid: [
      'Refined sugar & white flour',
      'Processed & fried foods',
      'Dairy (limit)',
      'Alcohol',
      'High-GI fruits (mango, grapes)',
      'Soy products',
    ],
    superfoods: [
      '🥦 Broccoli',
      '🫐 Blueberries',
      '🌿 Spearmint tea',
      '🐟 Salmon',
      '🥑 Avocado',
      '🌰 Walnuts',
      '🫘 Lentils',
      '🍃 Flaxseeds',
    ],
    weeklyPlan: [
      DayPlan(
        day: 'Monday',
        breakfast: Meal(
          name: 'Protein Oats Bowl',
          description: 'Rolled oats with nuts and berries',
          calories: '320 kcal',
          items: ['½ cup rolled oats', '1 tbsp chia seeds', 'Handful blueberries', '10 almonds', '1 tsp cinnamon', 'Unsweetened almond milk'],
        ),
        lunch: Meal(
          name: 'Quinoa Veggie Bowl',
          description: 'Quinoa with roasted vegetables',
          calories: '450 kcal',
          items: ['¾ cup quinoa', 'Roasted broccoli & bell peppers', '½ avocado', '2 tbsp olive oil dressing', 'Pumpkin seeds'],
        ),
        dinner: Meal(
          name: 'Baked Salmon & Greens',
          description: 'Omega-3 rich salmon with greens',
          calories: '480 kcal',
          items: ['150g baked salmon', '1 cup spinach sauté', '½ cup brown rice', 'Lemon & herb seasoning', 'Cherry tomatoes'],
        ),
        snack: Meal(
          name: 'Hormone-Balance Snack',
          description: 'Anti-inflammatory snack',
          calories: '150 kcal',
          items: ['1 tbsp almond butter', '1 apple (sliced)', 'Spearmint tea'],
        ),
      ),
      DayPlan(
        day: 'Tuesday',
        breakfast: Meal(
          name: 'Veggie Egg Scramble',
          description: 'Protein-rich egg scramble',
          calories: '290 kcal',
          items: ['2 whole eggs', 'Spinach & mushrooms', '½ tsp turmeric', '1 slice whole-grain toast', 'Green tea'],
        ),
        lunch: Meal(
          name: 'Lentil Soup & Salad',
          description: 'High-fiber lentil bowl',
          calories: '420 kcal',
          items: ['1 cup red lentil soup', 'Mixed greens salad', '1 tbsp flaxseeds', 'Olive oil & lemon dressing'],
        ),
        dinner: Meal(
          name: 'Chicken Stir-Fry',
          description: 'Lean protein with vegetables',
          calories: '460 kcal',
          items: ['150g grilled chicken', 'Stir-fried bok choy & broccoli', '½ cup cauliflower rice', 'Ginger-garlic sauce (no sugar)'],
        ),
        snack: Meal(
          name: 'Nut & Seed Mix',
          description: 'Hormone-balancing seeds',
          calories: '180 kcal',
          items: ['1 tbsp pumpkin seeds', '1 tbsp sunflower seeds', '5 walnuts', 'Chamomile tea'],
        ),
      ),
      DayPlan(
        day: 'Wednesday',
        breakfast: Meal(
          name: 'Green Smoothie',
          description: 'Detox green smoothie',
          calories: '280 kcal',
          items: ['1 cup spinach', '½ banana', '½ cup berries', '1 tbsp flaxseeds', 'Almond milk', '1 scoop protein powder'],
        ),
        lunch: Meal(
          name: 'Chickpea Salad Wrap',
          description: 'Plant protein wrap',
          calories: '430 kcal',
          items: ['½ cup chickpeas', 'Lettuce, cucumber, tomato', '1 whole-wheat wrap', '2 tbsp hummus', 'Olive oil drizzle'],
        ),
        dinner: Meal(
          name: 'Grilled Fish Tacos',
          description: 'Light & nutritious tacos',
          calories: '470 kcal',
          items: ['150g grilled tilapia', '2 corn tortillas', 'Cabbage slaw', 'Avocado slices', 'Salsa & lime'],
        ),
        snack: Meal(
          name: 'Anti-Inflammatory Bites',
          description: 'Turmeric golden milk',
          calories: '120 kcal',
          items: ['1 cup golden milk (turmeric + almond milk)', '1 tsp honey', '5 almonds'],
        ),
      ),
    ],
  ),

  DiseaseDietPlan(
    disease: 'Endometriosis',
    subtitle: 'Endometrial Tissue Disorder',
    description:
        'An anti-inflammatory, estrogen-reducing diet helps manage pain and slow abnormal tissue growth.',
    icon: Icons.favorite_border,
    primaryColor: const Color(0xFF7C4D9F),
    accentColor: const Color(0xFFF3E8FF),
    keyNutrients: [
      'Omega-3 Fatty Acids',
      'Antioxidants',
      'Vitamin E',
      'Iron',
      'B Vitamins',
      'Magnesium',
    ],
    foodsToAvoid: [
      'Red meat & processed meat',
      'Trans fats',
      'Caffeine (limit)',
      'Gluten (some benefit)',
      'Alcohol',
      'High-estrogen foods',
    ],
    superfoods: [
      '🫐 Berries (all types)',
      '🥬 Dark leafy greens',
      '🐟 Fatty fish',
      '🫚 Olive oil',
      '🍵 Green tea',
      '🧄 Garlic & turmeric',
      '🥕 Carrots & beets',
      '🌰 Brazil nuts',
    ],
    weeklyPlan: [
      DayPlan(
        day: 'Monday',
        breakfast: Meal(
          name: 'Anti-Inflammatory Bowl',
          description: 'Berry & seed power bowl',
          calories: '300 kcal',
          items: ['½ cup gluten-free oats', 'Mixed berries', '1 tbsp hemp seeds', '1 tsp turmeric', 'Coconut milk'],
        ),
        lunch: Meal(
          name: 'Mediterranean Plate',
          description: 'Estrogen-reducing plate',
          calories: '460 kcal',
          items: ['Grilled veggies (zucchini, eggplant)', '½ cup farro', '50g feta (optional)', 'Olive oil & lemon', 'Kalamata olives'],
        ),
        dinner: Meal(
          name: 'Turmeric Lentil Curry',
          description: 'Anti-inflammatory curry',
          calories: '490 kcal',
          items: ['1 cup green lentils', 'Turmeric, ginger, garlic', 'Coconut milk base', '½ cup brown rice', 'Fresh coriander'],
        ),
        snack: Meal(
          name: 'Antioxidant Snack',
          description: 'Free-radical fighting snack',
          calories: '140 kcal',
          items: ['1 cup mixed berries', '1 tbsp dark chocolate chips (70%+)', 'Green tea'],
        ),
      ),
      DayPlan(
        day: 'Tuesday',
        breakfast: Meal(
          name: 'Omega-3 Toast',
          description: 'Hormone-friendly breakfast',
          calories: '310 kcal',
          items: ['2 slices gluten-free bread', '½ avocado mashed', '1 tbsp smoked salmon', 'Lemon juice & dill', 'Herbal tea'],
        ),
        lunch: Meal(
          name: 'Kale Caesar Salad',
          description: 'Iron-rich leafy greens',
          calories: '440 kcal',
          items: ['2 cups kale', '½ cup chickpeas (roasted)', '2 tbsp tahini dressing', 'Nutritional yeast', 'Sunflower seeds'],
        ),
        dinner: Meal(
          name: 'Baked Mackerel',
          description: 'High omega-3 dinner',
          calories: '500 kcal',
          items: ['180g baked mackerel', 'Roasted beets & carrots', '½ cup quinoa', 'Lemon-herb dressing'],
        ),
        snack: Meal(
          name: 'Brazil Nut Mix',
          description: 'Selenium-rich snack',
          calories: '160 kcal',
          items: ['3 Brazil nuts', '1 tbsp pumpkin seeds', '½ cup raspberries'],
        ),
      ),
      DayPlan(
        day: 'Wednesday',
        breakfast: Meal(
          name: 'Chia Pudding',
          description: 'Overnight anti-inflammatory pudding',
          calories: '290 kcal',
          items: ['3 tbsp chia seeds', 'Coconut milk', 'Mango & kiwi topping', '1 tsp vanilla', 'Mint leaves'],
        ),
        lunch: Meal(
          name: 'Stuffed Bell Peppers',
          description: 'Colorful antioxidant lunch',
          calories: '450 kcal',
          items: ['2 bell peppers', 'Brown rice & black beans filling', 'Diced tomatoes', 'Cumin & paprika', 'Fresh parsley'],
        ),
        dinner: Meal(
          name: 'Ginger Salmon Stew',
          description: 'Soothing anti-pain stew',
          calories: '480 kcal',
          items: ['150g salmon chunks', 'Sweet potato & spinach', 'Fresh ginger & garlic', 'Vegetable broth', 'Turmeric'],
        ),
        snack: Meal(
          name: 'Calming Bites',
          description: 'Magnesium-rich snack',
          calories: '130 kcal',
          items: ['1 tbsp almond butter on celery', '1 tsp flaxseeds', 'Chamomile tea'],
        ),
      ),
    ],
  ),

  DiseaseDietPlan(
    disease: 'Thyroid Disorders',
    subtitle: 'Hypothyroidism & Hashimoto\'s',
    description:
        'A thyroid-supporting diet rich in iodine, selenium and zinc helps optimize thyroid hormone production.',
    icon: Icons.self_improvement,
    primaryColor: const Color(0xFF2E86AB),
    accentColor: const Color(0xFFE0F4FF),
    keyNutrients: [
      'Iodine',
      'Selenium',
      'Zinc',
      'Vitamin D',
      'Iron',
      'B12',
    ],
    foodsToAvoid: [
      'Raw cruciferous veggies (large amounts)',
      'Soy products',
      'Gluten (Hashimoto\'s)',
      'Highly processed foods',
      'Excess sugar',
      'Fluoride-heavy water',
    ],
    superfoods: [
      '🐟 Seaweed & kelp',
      '🥚 Eggs',
      '🫘 Brazil nuts (selenium)',
      '🍗 Lean chicken',
      '🫐 Berries',
      '🍠 Sweet potato',
      '🌱 Ashwagandha',
      '🧅 Cooked cruciferous veg',
    ],
    weeklyPlan: [
      DayPlan(
        day: 'Monday',
        breakfast: Meal(
          name: 'Thyroid Power Eggs',
          description: 'Selenium & iodine breakfast',
          calories: '310 kcal',
          items: ['2 eggs (scrambled)', '1 nori sheet (crumbled on top)', 'Sautéed mushrooms', '1 slice GF toast', '3 Brazil nuts', 'Black coffee or herbal tea'],
        ),
        lunch: Meal(
          name: 'Chicken & Sweet Potato',
          description: 'Energy-boosting balanced lunch',
          calories: '470 kcal',
          items: ['150g grilled chicken', '1 medium baked sweet potato', 'Steamed cooked broccoli', 'Olive oil & rosemary', 'Mixed greens salad'],
        ),
        dinner: Meal(
          name: 'Miso Soup & Rice',
          description: 'Iodine-rich Japanese dinner',
          calories: '440 kcal',
          items: ['1 bowl miso soup (seaweed, tofu, scallion)', '½ cup brown rice', '100g steamed fish', 'Pickled ginger', 'Edamame (small portion)'],
        ),
        snack: Meal(
          name: 'Energy Snack',
          description: 'Thyroid-boosting snack',
          calories: '170 kcal',
          items: ['1 boiled egg', '1 small orange', '3 Brazil nuts'],
        ),
      ),
      DayPlan(
        day: 'Tuesday',
        breakfast: Meal(
          name: 'Berry Protein Smoothie',
          description: 'Antioxidant morning smoothie',
          calories: '300 kcal',
          items: ['1 cup blueberries', '1 banana', '1 scoop vanilla protein', '1 tbsp flaxseeds', 'Almond milk', '1 tsp ashwagandha powder'],
        ),
        lunch: Meal(
          name: 'Tuna & Quinoa Bowl',
          description: 'Selenium-rich lunch bowl',
          calories: '480 kcal',
          items: ['1 can tuna (in water)', '½ cup quinoa', 'Cherry tomatoes & cucumber', 'Capers & lemon juice', '1 tbsp olive oil'],
        ),
        dinner: Meal(
          name: 'Lamb & Roasted Veg',
          description: 'Zinc-rich dinner',
          calories: '510 kcal',
          items: ['150g lean lamb', 'Roasted zucchini & bell peppers', '½ cup millet', 'Garlic & thyme', 'Side salad'],
        ),
        snack: Meal(
          name: 'Iodine Boost Snack',
          description: 'Sea-vegetable snack',
          calories: '100 kcal',
          items: ['1 pack roasted seaweed snacks', '5 almonds', 'Herbal tea'],
        ),
      ),
      DayPlan(
        day: 'Wednesday',
        breakfast: Meal(
          name: 'Oat & Seed Bowl',
          description: 'Iron-boosting breakfast',
          calories: '320 kcal',
          items: ['½ cup oats', '1 tbsp pumpkin seeds', '1 tbsp hemp seeds', 'Sliced banana', 'Fortified almond milk', 'Cinnamon'],
        ),
        lunch: Meal(
          name: 'Salmon & Lentil Salad',
          description: 'Complete thyroid-support meal',
          calories: '490 kcal',
          items: ['130g baked salmon', '½ cup puy lentils', 'Roasted beets', 'Arugula', 'Balsamic dressing'],
        ),
        dinner: Meal(
          name: 'Turkey & Veggie Stir-Fry',
          description: 'B12-rich lean protein dinner',
          calories: '460 kcal',
          items: ['150g ground turkey', 'Bok choy, snap peas, carrots', 'Tamari sauce (low sodium)', '½ cup rice noodles', 'Sesame seeds'],
        ),
        snack: Meal(
          name: 'Vitamin D Snack',
          description: 'Bone & thyroid support',
          calories: '150 kcal',
          items: ['1 cup fortified coconut yogurt', '1 tbsp sunflower seeds', 'Honey drizzle'],
        ),
      ),
    ],
  ),

  DiseaseDietPlan(
    disease: 'Diabetes',
    subtitle: 'Type 2 & Insulin Resistance',
    description:
        'A low-GI, high-fiber diet controls blood sugar spikes, improves insulin sensitivity and supports healthy weight.',
    icon: Icons.monitor_heart_outlined,
    primaryColor: const Color(0xFF2D9E6B),
    accentColor: const Color(0xFFE0F7EF),
    keyNutrients: [
      'Dietary Fiber',
      'Chromium',
      'Magnesium',
      'Alpha-Lipoic Acid',
      'Vitamin C',
      'Protein',
    ],
    foodsToAvoid: [
      'White bread, rice & pasta',
      'Sugary beverages & juices',
      'Sweets & desserts',
      'Fried & fast foods',
      'High-sugar fruits',
      'Alcohol',
    ],
    superfoods: [
      '🫘 Lentils & legumes',
      '🥦 Non-starchy vegetables',
      '🫐 Berries (low GI)',
      '🥜 Nuts & seeds',
      '🐟 Fatty fish',
      '🍃 Bitter gourd',
      '🧄 Cinnamon & fenugreek',
      '🌾 Whole grains',
    ],
    weeklyPlan: [
      DayPlan(
        day: 'Monday',
        breakfast: Meal(
          name: 'Low-GI Breakfast',
          description: 'Blood sugar stable start',
          calories: '290 kcal',
          items: ['2 boiled eggs', '1 slice whole-grain toast', '½ avocado', 'Sliced tomato & cucumber', 'Unsweetened green tea'],
        ),
        lunch: Meal(
          name: 'Fiber-Rich Plate',
          description: 'Balanced glycemic lunch',
          calories: '440 kcal',
          items: ['½ cup black beans', '½ cup brown rice', 'Sautéed spinach & peppers', '2 tbsp salsa', 'Lime squeeze'],
        ),
        dinner: Meal(
          name: 'Grilled Chicken & Veg',
          description: 'Lean protein dinner',
          calories: '460 kcal',
          items: ['150g grilled chicken breast', 'Steamed broccoli, cauliflower & carrots', '½ cup quinoa', 'Lemon-herb dressing'],
        ),
        snack: Meal(
          name: 'Stable Sugar Snack',
          description: 'Low-GI afternoon snack',
          calories: '130 kcal',
          items: ['1 small apple', '1 tbsp peanut butter (no sugar added)', 'Cinnamon sprinkle'],
        ),
      ),
      DayPlan(
        day: 'Tuesday',
        breakfast: Meal(
          name: 'Greek Yogurt Parfait',
          description: 'Protein-rich, low-sugar',
          calories: '280 kcal',
          items: ['¾ cup plain Greek yogurt (unsweetened)', '¼ cup strawberries', '1 tbsp chia seeds', '1 tbsp walnuts', 'Stevia (optional)'],
        ),
        lunch: Meal(
          name: 'Chickpea Power Bowl',
          description: 'Plant-based fiber bowl',
          calories: '450 kcal',
          items: ['¾ cup roasted chickpeas', 'Mixed greens & cucumber', 'Diced red onion & tomato', '2 tbsp tahini dressing', '1 tbsp pumpkin seeds'],
        ),
        dinner: Meal(
          name: 'Baked Cod & Greens',
          description: 'Lean omega-3 dinner',
          calories: '470 kcal',
          items: ['180g baked cod', 'Steamed kale & asparagus', '½ cup barley', 'Garlic & olive oil', 'Cherry tomatoes'],
        ),
        snack: Meal(
          name: 'Protein Snack',
          description: 'Insulin-stabilizing snack',
          calories: '120 kcal',
          items: ['1 boiled egg', '5 almonds', 'Cucumber slices', 'Herbal tea'],
        ),
      ),
      DayPlan(
        day: 'Wednesday',
        breakfast: Meal(
          name: 'Fenugreek Oats',
          description: 'Blood-sugar regulating oats',
          calories: '300 kcal',
          items: ['½ cup steel-cut oats', '1 tsp fenugreek powder', '½ cup blueberries', '1 tbsp flaxseeds', 'Unsweetened almond milk'],
        ),
        lunch: Meal(
          name: 'Bitter Gourd Stir-Fry',
          description: 'Traditional sugar-control meal',
          calories: '380 kcal',
          items: ['1 cup bitter gourd (karela)', 'Scrambled eggs', 'Onion & tomato', '1 chapati (whole wheat)', 'Dal (lentil soup)'],
        ),
        dinner: Meal(
          name: 'Salmon & Veggie Bake',
          description: 'Anti-inflammatory dinner',
          calories: '490 kcal',
          items: ['150g salmon', 'Roasted Brussels sprouts', 'Roasted bell peppers', '½ cup lentils', 'Dill & lemon'],
        ),
        snack: Meal(
          name: 'Cinnamon Snack',
          description: 'Natural blood sugar support',
          calories: '110 kcal',
          items: ['1 cup cinnamon tea', '1 small pear', '3 walnut halves'],
        ),
      ),
    ],
  ),
];


class DietPlanPage extends StatefulWidget {
  const DietPlanPage({Key? key}) : super(key: key);

  @override
  State<DietPlanPage> createState() => _DietPlanPageState();
}

class _DietPlanPageState extends State<DietPlanPage>
    with TickerProviderStateMixin {
      String dietPlan = "";
  bool isLoading = true;
  int _selectedDiseaseIndex = 0;
  int _selectedDayIndex = 0;
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _tabController.addListener(() {
      if (!_tabController.indexIsChanging) {
        setState(() {
          _selectedDiseaseIndex = _tabController.index;
          _selectedDayIndex = 0;
        });
      }
    });
    fetchDietPlan();
  }
  Future<void> fetchDietPlan() async {

  setState(() {
    isLoading = true;
  });

  String result = await GroqService().generateDietPlan(
  "irregular periods, hormonal imbalance");

  setState(() {
    dietPlan = result;
    isLoading = false;
  });

}

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  DiseaseDietPlan get _currentPlan => dietPlans[_selectedDiseaseIndex];
  DayPlan get _currentDay => _currentPlan.weeklyPlan[_selectedDayIndex];

  

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildDiseaseTabBar(),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    _buildDietOverviewCard(),
                    _buildSuperFoodsSection(),
                    _buildFoodsToAvoidSection(),
                    _buildWeeklyPlanSection(),
                    _buildNutrientsSection(),
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

  // ── Header ──────────────────────────────────
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
              const Text(
                'Diet Plan Guide',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                ),
              ),
              Text(
                'Personalized nutrition for women\'s health',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.85),
                  fontSize: 12,
                ),
              ),
            ],
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(_currentPlan.icon, color: Colors.white, size: 26),
          ),
        ],
      ),
    );
  }

  // ── Disease Tab Bar ──────────────────────────
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
        labelStyle: const TextStyle(
          fontWeight: FontWeight.w700,
          fontSize: 13,
        ),
        unselectedLabelStyle: const TextStyle(
          fontWeight: FontWeight.w500,
          fontSize: 12,
        ),
        tabs: dietPlans.map((plan) {
          return Tab(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Text(plan.disease),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Overview Card ────────────────────────────
  Widget _buildDietOverviewCard() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _currentPlan.primaryColor,
            _currentPlan.primaryColor.withOpacity(0.75),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: _currentPlan.primaryColor.withOpacity(0.3),
            blurRadius: 15,
            offset: const Offset(0, 5),
          ),
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
                  color: Colors.white.withOpacity(0.25),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(_currentPlan.icon, color: Colors.white, size: 28),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _currentPlan.disease,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text(
                      _currentPlan.subtitle,
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.85),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              _currentPlan.description,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13.5,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Superfoods ───────────────────────────────
  Widget _buildSuperFoodsSection() {
    return _buildSection(
      title: '⭐ Superfoods to Include',
      color: _currentPlan.primaryColor,
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: _currentPlan.superfoods.map((food) {
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: _currentPlan.accentColor,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _currentPlan.primaryColor.withOpacity(0.3),
              ),
            ),
            child: Text(
              food,
              style: TextStyle(
                color: _currentPlan.primaryColor,
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Foods to Avoid ───────────────────────────
  Widget _buildFoodsToAvoidSection() {
    return _buildSection(
      title: '🚫 Foods to Avoid',
      color: _currentPlan.primaryColor,
      child: Column(
        children: _currentPlan.foodsToAvoid.map((food) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: const BoxDecoration(
                    color: Color(0xFFE53E3E),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    food,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Color(0xFF444444),
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Weekly Plan Section ──────────────────────
  Widget _buildWeeklyPlanSection() {
    return _buildSection(
      title: '📅 Weekly Meal Plan',
      color: _currentPlan.primaryColor,
      child: Column(
        children: [
          // Day selector
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
                    padding: const EdgeInsets.symmetric(
                        horizontal: 18, vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? _currentPlan.primaryColor
                          : Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: _currentPlan.primaryColor,
                        width: 1.5,
                      ),
                    ),
                    child: Text(
                      _currentPlan.weeklyPlan[index].day,
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
          // Meal cards
          _buildMealCard(
            '🌅 Breakfast',
            _currentDay.breakfast,
            const Color(0xFFFFF3CD),
            const Color(0xFFD97706),
          ),
          const SizedBox(height: 12),
          _buildMealCard(
            '☀️ Lunch',
            _currentDay.lunch,
            const Color(0xFFD1FAE5),
            const Color(0xFF059669),
          ),
          const SizedBox(height: 12),
          _buildMealCard(
            '🌙 Dinner',
            _currentDay.dinner,
            const Color(0xFFEDE9FE),
            const Color(0xFF7C3AED),
          ),
          const SizedBox(height: 12),
          _buildMealCard(
            '🍎 Snack',
            _currentDay.snack,
            const Color(0xFFFFE4E6),
            const Color(0xFFE11D48),
          ),
        ],
      ),
    );
  }

  Widget _buildMealCard(
      String label, Meal meal, Color bgColor, Color accentColor) {
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
            decoration: BoxDecoration(
              color: accentColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(label.split(' ')[0], style: const TextStyle(fontSize: 20)),
          ),
          title: Text(
            meal.name,
            style: TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 15,
              color: accentColor,
            ),
          ),
          subtitle: Row(
            children: [
              Text(
                meal.description,
                style: TextStyle(
                  fontSize: 12,
                  color: accentColor.withOpacity(0.8),
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  meal.calories,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: accentColor,
                  ),
                ),
              ),
            ],
          ),
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Ingredients & Portions:',
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    color: accentColor,
                  ),
                ),
                const SizedBox(height: 8),
                ...meal.items.map((item) => Padding(
                      padding: const EdgeInsets.only(bottom: 5),
                      child: Row(
                        children: [
                          Icon(Icons.check_circle_outline,
                              size: 16, color: accentColor),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              item,
                              style: TextStyle(
                                fontSize: 13,
                                color: accentColor.withOpacity(0.9),
                              ),
                            ),
                          ),
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

  // ── Key Nutrients Section ────────────────────
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
        children: _currentPlan.keyNutrients.map((nutrient) {
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: _currentPlan.accentColor,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _currentPlan.primaryColor.withOpacity(0.2),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.local_pharmacy_outlined,
                  size: 16,
                  color: _currentPlan.primaryColor,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    nutrient,
                    style: TextStyle(
                      color: _currentPlan.primaryColor,
                      fontWeight: FontWeight.w600,
                      fontSize: 12,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Generic Section Wrapper ──────────────────
  Widget _buildSection({
    required String title,
    required Color color,
    required Widget child,
  }) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 4,
                height: 20,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 10),
              Text(
                title,
                style: TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.w800,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}