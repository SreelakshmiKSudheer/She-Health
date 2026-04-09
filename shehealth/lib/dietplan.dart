import 'package:flutter/material.dart';
import 'services/groq_service.dart';



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

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────

Meal _m(String name, String desc, String cal, List<String> items) =>
    Meal(name: name, description: desc, calories: cal, items: items);

Exercise _ex(String name, String dur, String intensity, String benefit, IconData icon, List<String> steps) =>
    Exercise(name: name, duration: dur, intensity: intensity, benefit: benefit, icon: icon, steps: steps);

// ─────────────────────────────────────────────
//  PCOS DIET DATA (7 days)
// ─────────────────────────────────────────────

final List<DayPlan> _pcosMeals = [
  DayPlan(day: 'Monday',
    breakfast: _m('Protein Oats Bowl','Rolled oats with nuts and berries','320 kcal',['½ cup rolled oats','1 tbsp chia seeds','Handful blueberries','10 almonds','1 tsp cinnamon','Unsweetened almond milk']),
    lunch: _m('Quinoa Veggie Bowl','Quinoa with roasted vegetables','450 kcal',['¾ cup quinoa','Roasted broccoli & bell peppers','½ avocado','2 tbsp olive oil dressing','Pumpkin seeds']),
    dinner: _m('Baked Salmon & Greens','Omega-3 rich salmon with greens','480 kcal',['150g baked salmon','1 cup spinach sauté','½ cup brown rice','Lemon & herb seasoning','Cherry tomatoes']),
    snack: _m('Hormone-Balance Snack','Anti-inflammatory snack','150 kcal',['1 tbsp almond butter','1 apple (sliced)','Spearmint tea']),
  ),
  DayPlan(day: 'Tuesday',
    breakfast: _m('Veggie Egg Scramble','Protein-rich egg scramble','290 kcal',['2 whole eggs','Spinach & mushrooms','½ tsp turmeric','1 slice whole-grain toast','Green tea']),
    lunch: _m('Lentil Soup & Salad','High-fiber lentil bowl','420 kcal',['1 cup red lentil soup','Mixed greens salad','1 tbsp flaxseeds','Olive oil & lemon dressing']),
    dinner: _m('Chicken Stir-Fry','Lean protein with vegetables','460 kcal',['150g grilled chicken','Stir-fried bok choy & broccoli','½ cup cauliflower rice','Ginger-garlic sauce (no sugar)']),
    snack: _m('Nut & Seed Mix','Hormone-balancing seeds','180 kcal',['1 tbsp pumpkin seeds','1 tbsp sunflower seeds','5 walnuts','Chamomile tea']),
  ),
  DayPlan(day: 'Wednesday',
    breakfast: _m('Green Smoothie','Detox green smoothie','280 kcal',['1 cup spinach','½ banana','½ cup berries','1 tbsp flaxseeds','Almond milk','1 scoop protein powder']),
    lunch: _m('Chickpea Salad Wrap','Plant protein wrap','430 kcal',['½ cup chickpeas','Lettuce, cucumber, tomato','1 whole-wheat wrap','2 tbsp hummus','Olive oil drizzle']),
    dinner: _m('Grilled Fish Tacos','Light & nutritious tacos','470 kcal',['150g grilled tilapia','2 corn tortillas','Cabbage slaw','Avocado slices','Salsa & lime']),
    snack: _m('Golden Milk','Anti-inflammatory turmeric drink','120 kcal',['1 cup golden milk (turmeric + almond milk)','1 tsp honey','5 almonds']),
  ),
  DayPlan(day: 'Thursday',
    breakfast: _m('Chia Berry Parfait','Antioxidant layered parfait','300 kcal',['3 tbsp chia seeds soaked overnight','Coconut yogurt','½ cup mixed berries','1 tbsp walnuts','1 tsp maple syrup']),
    lunch: _m('Brown Rice & Bean Bowl','Low-GI fiber-rich bowl','440 kcal',['½ cup brown rice','½ cup black beans','Diced tomato & cucumber','Lime juice & cumin','Fresh coriander']),
    dinner: _m('Turkey Lettuce Wraps','Low-carb lean protein dinner','420 kcal',['150g ground turkey','Butter lettuce leaves','Diced water chestnuts & scallion','Tamari sauce','Sesame seeds']),
    snack: _m('Spearmint Nut Snack','Hormone-supportive snack','140 kcal',['1 cup spearmint tea','10 cashews','1 small pear']),
  ),
  DayPlan(day: 'Friday',
    breakfast: _m('Avocado Toast & Eggs','Healthy fats & protein start','350 kcal',['2 slices rye bread','½ mashed avocado','2 poached eggs','Chili flakes & lemon','Herbal tea']),
    lunch: _m('Spinach Walnut Salad','Magnesium-rich large salad','400 kcal',['2 cups baby spinach','¼ cup walnuts','Sliced beets','Goat cheese (small amount)','Balsamic vinaigrette']),
    dinner: _m('Baked Cod & Vegetables','Light omega-3 dinner','460 kcal',['160g baked cod fillet','Roasted zucchini & asparagus','½ cup quinoa','Lemon-dill sauce','Side salad']),
    snack: _m('Flaxseed Smoothie','Hormone-balancing smoothie','160 kcal',['1 cup almond milk','1 tbsp flaxseeds','½ banana','1 tbsp almond butter']),
  ),
  DayPlan(day: 'Saturday',
    breakfast: _m('Sweet Potato Pancakes','Low-GI weekend breakfast','330 kcal',['½ cup sweet potato (mashed)','2 eggs','2 tbsp oat flour','Cinnamon & vanilla','Fresh berries on top']),
    lunch: _m('Grilled Veggie Platter','Colorful antioxidant lunch','420 kcal',['Grilled eggplant, zucchini, bell peppers','½ cup hummus','1 whole-wheat pita','Olive oil & herbs','Cherry tomatoes']),
    dinner: _m('Lamb & Cauliflower Mash','Zinc-rich satisfying dinner','490 kcal',['130g lean lamb','Cauliflower mash (olive oil + garlic)','Steamed green beans','Mint sauce (no sugar)','Mixed herb salad']),
    snack: _m('Dark Chocolate Bites','Magnesium-rich treat','130 kcal',['3 squares 85% dark chocolate','10 raspberries','Green tea']),
  ),
  DayPlan(day: 'Sunday',
    breakfast: _m('Overnight Oats','Prep-ahead balanced breakfast','310 kcal',['½ cup rolled oats','1 cup almond milk','1 tbsp flaxseeds','Sliced kiwi & mango','1 tsp cinnamon']),
    lunch: _m('Tuna Nicoise Salad','Protein & omega-3 rich lunch','460 kcal',['1 can tuna in water','Boiled egg','Green beans, olives, tomato','1 tbsp olive oil & lemon','Mixed lettuce']),
    dinner: _m('Vegetable Dhal','Warming Indian-spiced dinner','450 kcal',['1 cup red lentil dhal','Spinach & tomato base','Turmeric, cumin & garam masala','1 whole-wheat roti','Cucumber raita (low-fat)']),
    snack: _m('Zinc-Boost Snack','Seed-rich recovery snack','150 kcal',['2 tbsp pumpkin seeds','1 tbsp hemp seeds','1 small apple','Chamomile tea']),
  ),
];

// ─────────────────────────────────────────────
//  PCOS WORKOUT DATA (7 days)
// ─────────────────────────────────────────────

final List<WorkoutDay> _pcosWorkout = [
  WorkoutDay(day: 'Monday', focus: 'Strength Training', exercises: [
    _ex('Squats','3 × 12 reps','Moderate','Improves insulin sensitivity & leg strength',Icons.fitness_center,['Stand feet shoulder-width apart','Lower hips as if sitting back into a chair','Keep knees behind toes, back straight','Push through heels to return to standing','Rest 60 seconds between sets']),
    _ex('Dumbbell Deadlifts','3 × 10 reps','Moderate','Builds metabolic muscle, regulates hormones',Icons.fitness_center,['Hold dumbbells in front of thighs','Hinge at hips, lower weights down shins','Keep back flat, core tight','Drive hips forward to stand','Squeeze glutes at the top']),
    _ex('Plank Hold','3 × 30 sec','Moderate','Core strength reduces belly fat linked to PCOS',Icons.self_improvement,['Start in forearm plank position','Keep body in a straight line head to heel','Engage core & glutes','Breathe steadily, don\'t hold breath','Build up to 60 seconds over time']),
  ]),
  WorkoutDay(day: 'Tuesday', focus: 'Low-Impact Cardio + Yoga', exercises: [
    _ex('Brisk Walking','30 min','Low','Reduces insulin resistance, gentle on joints',Icons.directions_walk,['Walk at a pace where you can still talk','Swing arms naturally for balance','Maintain upright posture','Aim for 100 steps/min','Cool down with 5 min slow walk']),
    _ex('Butterfly Pose','5 min hold','Low','Stimulates ovaries, relieves pelvic tension',Icons.self_improvement,['Sit on floor, bring soles of feet together','Let knees fall open to sides','Hold feet with both hands','Breathe deeply, gently press knees down','Hold and release tension with each exhale']),
    _ex('Cat-Cow Stretch','10 rounds','Low','Massages reproductive organs, eases cramps',Icons.self_improvement,['Start on hands and knees (tabletop)','Inhale: drop belly, lift chest (cow)','Exhale: round spine toward ceiling (cat)','Move slowly, syncing breath with movement','Repeat fluidly 10 times']),
  ]),
  WorkoutDay(day: 'Wednesday', focus: 'HIIT + Core', exercises: [
    _ex('Jump Squats','4 × 30 sec','High','Boosts metabolism & burns visceral fat',Icons.directions_run,['Start in squat position','Jump explosively upward','Land softly with bent knees','Immediately lower back into squat','Rest 30 sec between sets']),
    _ex('Mountain Climbers','3 × 30 sec','High','Full-body fat burn, improves core stability',Icons.directions_run,['Start in high plank position','Drive right knee toward chest','Switch legs rapidly in running motion','Keep hips level, core tight','Breathe rhythmically throughout']),
    _ex('Bicycle Crunches','3 × 15 reps','Moderate','Tones core, reduces abdominal fat',Icons.fitness_center,['Lie on back, hands behind head','Lift shoulders off ground','Bring left elbow to right knee','Simultaneously extend left leg','Alternate sides in a pedalling motion']),
  ]),
  WorkoutDay(day: 'Thursday', focus: 'Active Recovery & Stretching', exercises: [
    _ex('Restorative Walk','20–30 min','Low','Lowers cortisol, supports hormonal balance',Icons.self_improvement,['Walk slowly in nature','Focus on deep belly breathing','Release all muscle tension consciously','Avoid any strenuous activity today','Hydrate well and rest fully']),
    _ex('Seated Forward Fold','5 min hold','Low','Calms nervous system, stretches lower back',Icons.self_improvement,['Sit with legs extended forward','Inhale to lengthen spine','Exhale and fold forward gently from hips','Hold shins or ankles — never force','Breathe into the stretch for 5 minutes']),
    _ex('Legs Up the Wall','10 min','Low','Improves pelvic circulation, reduces fatigue',Icons.self_improvement,['Sit sideways against a wall','Swing legs up as you lie down','Rest legs vertically against wall','Place hands on belly, breathe slowly','Stay 10 min with eyes closed']),
  ]),
  WorkoutDay(day: 'Friday', focus: 'Strength + Resistance', exercises: [
    _ex('Lunges','3 × 12 each leg','Moderate','Builds lower body strength, improves insulin uptake',Icons.fitness_center,['Stand tall, hands on hips','Step right foot forward into lunge','Lower back knee toward floor','Push through front heel to return','Alternate legs, keep torso upright']),
    _ex('Push-Ups','3 × 10 reps','Moderate','Upper body strength, increases resting metabolism',Icons.fitness_center,['Start in high plank position','Lower chest to just above floor','Keep elbows at 45° to body','Push back up fully','Modify on knees if needed']),
    _ex('Glute Bridges','3 × 15 reps','Moderate','Activates glutes, reduces hip & pelvic tension',Icons.fitness_center,['Lie on back, feet hip-width apart','Press feet down and lift hips','Squeeze glutes at the top','Hold 2 seconds, lower slowly','Keep core engaged throughout']),
  ]),
  WorkoutDay(day: 'Saturday', focus: 'Cardio Dance / Swimming', exercises: [
    _ex('Dance Cardio / Zumba','30–40 min','Moderate','Fun cardio that reduces stress hormones',Icons.music_note,['Play upbeat music and move freely','Focus on full-body movement','No rules — just keep moving','Let go of tension while dancing','End with slow stretches']),
    _ex('Swimming','25 min','Moderate','Full-body low-impact cardio, reduces inflammation',Icons.pool,['Warm up with 2 min easy freestyle','Swim at moderate pace for 20 min','Mix strokes: freestyle, breaststroke','Focus on rhythmic breathing','Cool down with gentle floating stretch']),
    _ex('Hip Circles','3 min','Low','Loosens pelvic region, improves ovarian blood flow',Icons.self_improvement,['Stand feet shoulder-width apart','Place hands on hips','Draw large circles with hips clockwise','Repeat counter-clockwise','10 slow circles each direction']),
  ]),
  WorkoutDay(day: 'Sunday', focus: 'Rest & Meditation', exercises: [
    _ex('Full Rest Day','All day','Low','Recovery essential for hormone regulation',Icons.nights_stay,['No intense exercise today','Take a gentle 10-min walk if desired','Prioritise 7–9 hours of sleep','Practice gratitude journaling','Prepare meals and mindset for the week ahead']),
    _ex('Guided Meditation','15 min','Low','Reduces cortisol & anxiety that disrupt hormones',Icons.psychology,['Find a quiet, comfortable space','Sit or lie down comfortably','Close eyes, focus on slow deep breathing','Use a guided app or simply follow your breath','Let thoughts pass without attachment']),
    _ex('Progressive Muscle Relaxation','10 min','Low','Releases stored physical tension, improves sleep',Icons.self_improvement,['Lie down in savasana position','Tense each muscle group for 5 seconds','Release fully and notice the difference','Work from toes to head systematically','End with 5 deep belly breaths']),
  ]),
];

// ─────────────────────────────────────────────
//  ENDOMETRIOSIS DATA (7 days)
// ─────────────────────────────────────────────

final List<DayPlan> _endoMeals = [
  DayPlan(day: 'Monday',
    breakfast: _m('Anti-Inflammatory Bowl','Berry & seed power bowl','300 kcal',['½ cup gluten-free oats','Mixed berries','1 tbsp hemp seeds','1 tsp turmeric','Coconut milk']),
    lunch: _m('Mediterranean Plate','Estrogen-reducing plate','460 kcal',['Grilled veggies (zucchini, eggplant)','½ cup farro','50g feta (optional)','Olive oil & lemon','Kalamata olives']),
    dinner: _m('Turmeric Lentil Curry','Anti-inflammatory curry','490 kcal',['1 cup green lentils','Turmeric, ginger, garlic','Coconut milk base','½ cup brown rice','Fresh coriander']),
    snack: _m('Antioxidant Snack','Free-radical fighting snack','140 kcal',['1 cup mixed berries','1 tbsp dark chocolate chips (70%+)','Green tea']),
  ),
  DayPlan(day: 'Tuesday',
    breakfast: _m('Omega-3 Toast','Hormone-friendly breakfast','310 kcal',['2 slices gluten-free bread','½ avocado mashed','1 tbsp smoked salmon','Lemon juice & dill','Herbal tea']),
    lunch: _m('Kale Caesar Salad','Iron-rich leafy greens','440 kcal',['2 cups kale','½ cup chickpeas (roasted)','2 tbsp tahini dressing','Nutritional yeast','Sunflower seeds']),
    dinner: _m('Baked Mackerel','High omega-3 dinner','500 kcal',['180g baked mackerel','Roasted beets & carrots','½ cup quinoa','Lemon-herb dressing']),
    snack: _m('Brazil Nut Mix','Selenium-rich snack','160 kcal',['3 Brazil nuts','1 tbsp pumpkin seeds','½ cup raspberries']),
  ),
  DayPlan(day: 'Wednesday',
    breakfast: _m('Chia Pudding','Overnight anti-inflammatory pudding','290 kcal',['3 tbsp chia seeds','Coconut milk','Mango & kiwi topping','1 tsp vanilla','Mint leaves']),
    lunch: _m('Stuffed Bell Peppers','Colorful antioxidant lunch','450 kcal',['2 bell peppers','Brown rice & black beans filling','Diced tomatoes','Cumin & paprika','Fresh parsley']),
    dinner: _m('Ginger Salmon Stew','Soothing anti-pain stew','480 kcal',['150g salmon chunks','Sweet potato & spinach','Fresh ginger & garlic','Vegetable broth','Turmeric']),
    snack: _m('Calming Bites','Magnesium-rich snack','130 kcal',['1 tbsp almond butter on celery','1 tsp flaxseeds','Chamomile tea']),
  ),
  DayPlan(day: 'Thursday',
    breakfast: _m('Iron-Boost Smoothie','Iron & folate morning blend','270 kcal',['1 cup spinach','½ cup frozen berries','1 tbsp molasses','1 cup fortified oat milk','1 tbsp hemp seeds']),
    lunch: _m('Sardine & Avocado Toast','Omega-3 rich quick lunch','420 kcal',['2 slices GF bread','1 can sardines in olive oil','½ mashed avocado','Lemon juice & capers','Cucumber slices']),
    dinner: _m('Veggie Thai Curry','Anti-inflammatory Thai spice dinner','470 kcal',['1 cup chickpeas','Thai green curry paste (low-sodium)','Coconut milk','Bok choy, broccoli & snap peas','½ cup jasmine rice']),
    snack: _m('Vitamin E Snack','Anti-pain vitamin-rich snack','150 kcal',['1 tbsp sunflower seeds','½ cup papaya','5 almonds','Green tea']),
  ),
  DayPlan(day: 'Friday',
    breakfast: _m('Turmeric Scrambled Eggs','Anti-inflammatory protein breakfast','300 kcal',['2 eggs scrambled with turmeric','Baby spinach & cherry tomatoes','1 slice GF toast','1 tsp olive oil','Herbal tea']),
    lunch: _m('Rainbow Quinoa Salad','Antioxidant-packed colorful salad','430 kcal',['½ cup quinoa','Shredded purple cabbage, grated carrot, diced mango','1 tbsp pumpkin seeds','Lime & ginger dressing','Fresh mint']),
    dinner: _m('Herb-Baked Chicken','Lean protein with herbs','460 kcal',['160g baked chicken breast','Rosemary, thyme & garlic','Roasted sweet potato wedges','Steamed asparagus','Lemon drizzle']),
    snack: _m('Berry Kefir Cup','Probiotic gut-support snack','140 kcal',['½ cup coconut kefir (dairy-free)','½ cup blueberries','1 tsp flaxseeds']),
  ),
  DayPlan(day: 'Saturday',
    breakfast: _m('Coconut Mango Smoothie','Tropical B-vitamin blend','280 kcal',['1 cup coconut milk','½ cup frozen mango','½ banana','1 tbsp hemp seeds','1 tsp turmeric','Pinch of black pepper']),
    lunch: _m('Lentil & Roasted Beet Salad','Iron & antioxidant rich bowl','450 kcal',['½ cup puy lentils','2 roasted beets','2 cups arugula','¼ cup walnuts','Balsamic vinegar dressing']),
    dinner: _m('Cod en Papillote','Delicate baked fish with greens','450 kcal',['160g cod fillet','Julienned zucchini & carrots','Lemon slices & fresh dill','Olive oil drizzle','Side of steamed quinoa']),
    snack: _m('Dark Chocolate & Walnut','Magnesium & omega-3 treat','150 kcal',['2 squares 85% dark chocolate','6 walnuts','Spearmint tea']),
  ),
  DayPlan(day: 'Sunday',
    breakfast: _m('Warm Spiced Porridge','Comforting anti-inflammatory breakfast','310 kcal',['½ cup GF oats','Cinnamon, ginger & cardamom','1 tbsp chia seeds','Stewed apple','Oat milk']),
    lunch: _m('Roasted Veggie Grain Bowl','Nourishing end-of-week bowl','460 kcal',['½ cup farro or buckwheat','Roasted butternut squash & red onion','Handful baby spinach','Tahini lemon dressing','Toasted pumpkin seeds']),
    dinner: _m('Salmon & Lentil Dal','Omega-3 meets iron dinner','490 kcal',['150g baked salmon','½ cup red lentil dal','Garlic, ginger & turmeric base','Wilted kale','Lemon wedge']),
    snack: _m('Rest Day Snack','Calming end-of-week snack','120 kcal',['1 cup chamomile tea','3 Brazil nuts','½ cup blueberries']),
  ),
];

final List<WorkoutDay> _endoWorkout = [
  WorkoutDay(day: 'Monday', focus: 'Restorative Yoga', exercises: [
    _ex('Child\'s Pose (Balasana)','5 min hold','Low','Relieves pelvic tension & lower back pain',Icons.self_improvement,['Kneel and sit back on heels','Stretch arms forward on the mat','Rest forehead gently on mat','Breathe deeply, expanding back with each inhale','Hold and release tension with every exhale']),
    _ex('Legs-Up-the-Wall','10 min','Low','Reduces pelvic inflammation & improves circulation',Icons.self_improvement,['Sit sideways next to a wall','Swing legs up as you lie down','Rest legs against the wall at 90°','Place hands on belly, breathe slowly','Stay for 10 min, close your eyes']),
    _ex('Seated Forward Fold','5 min hold','Low','Gently stretches pelvic floor & lower back',Icons.self_improvement,['Sit on floor, legs extended forward','Inhale to lengthen spine','Exhale and fold forward from the hips','Hold ankles or shins, never force','Breathe into the stretch for 5 minutes']),
  ]),
  WorkoutDay(day: 'Tuesday', focus: 'Gentle Walking & Breathing', exercises: [
    _ex('Mindful Walking','30 min','Low','Reduces cortisol & systemic inflammation',Icons.directions_walk,['Walk at a comfortable, gentle pace','Focus on breathing — inhale 4 steps, exhale 4 steps','Notice surroundings, stay present','Choose flat, even terrain','End with 5 min of gentle stretching']),
    _ex('Diaphragmatic Breathing','10 min','Low','Activates parasympathetic system, reduces pain',Icons.air,['Lie down or sit comfortably','Place one hand on chest, one on belly','Inhale through nose, let belly rise (not chest)','Exhale slowly through pursed lips','Repeat for 10 minutes daily']),
    _ex('Gentle Hip Flexor Stretch','3 min each side','Low','Releases pelvic tension from adhesions',Icons.self_improvement,['Kneel on right knee, left foot forward','Shift hips forward gently until stretch felt','Keep torso upright, core lightly engaged','Hold 3 min, breathe deeply','Repeat on other side']),
  ]),
  WorkoutDay(day: 'Wednesday', focus: 'Pilates + Core', exercises: [
    _ex('Pelvic Tilts','3 × 15 reps','Low','Strengthens pelvic floor, eases endometriosis pain',Icons.fitness_center,['Lie on back, knees bent, feet flat','Gently flatten lower back into the floor','Tilt pelvis upward slightly','Hold 3 seconds, then release','Breathe naturally throughout']),
    _ex('Pilates Hundred','3 × 10 breath cycles','Moderate','Builds deep core without stressing pelvic area',Icons.fitness_center,['Lie on back, raise legs to tabletop','Lift head and shoulders off mat','Arms by sides, pump arms up and down','Inhale for 5 pumps, exhale for 5 pumps','Keep lower back pressed to mat']),
    _ex('Glute Bridges','3 × 12 reps','Low','Strengthens glutes & reduces pelvic floor tension',Icons.fitness_center,['Lie on back, feet hip-width apart','Press feet into floor and lift hips','Squeeze glutes at the top','Hold 2 seconds, lower slowly','Keep core engaged throughout']),
  ]),
  WorkoutDay(day: 'Thursday', focus: 'Swimming / Hydrotherapy', exercises: [
    _ex('Slow Swimming','25–30 min','Low','Full-body cardio without pelvic strain',Icons.pool,['Choose breaststroke or backstroke','Focus on slow, controlled movements','Breathe rhythmically with each stroke','Avoid flip turns if they cause pain','Stretch gently in the water after']),
    _ex('Water Walking','15 min','Low','Resistance without joint stress, soothes inflammation',Icons.pool,['Walk in pool at waist depth','Swing arms naturally through the water','Focus on upright posture','Alternate forward and backward walking','End with gentle pool-side stretches']),
    _ex('Aqua Hip Circles','5 min','Low','Gently mobilises pelvic joints',Icons.self_improvement,['Stand in waist-deep water','Place hands on pool edge for balance','Draw slow large circles with hips','10 circles clockwise, 10 counter-clockwise','Move slowly and breathe deeply']),
  ]),
  WorkoutDay(day: 'Friday', focus: 'Yin Yoga', exercises: [
    _ex('Dragon Pose (Low Lunge)','5 min each side','Low','Deep hip flexor stretch, releases stored tension',Icons.self_improvement,['Step right foot forward into low lunge','Lower left knee to mat','Sink hips gently forward','Arms rest on front knee or floor','Hold 5 min each side']),
    _ex('Sleeping Swan (Pigeon Pose)','5 min each side','Low','Deeply opens hip rotators, relieves pelvic tightness',Icons.self_improvement,['From downward dog, bring right knee to right wrist','Extend left leg back on the mat','Fold forward over bent leg','Use props (blanket) under hip if needed','Hold 5 min each side']),
    _ex('Supine Twist','3 min each side','Low','Massages abdominal organs, reduces bloating',Icons.self_improvement,['Lie on back, draw right knee to chest','Guide knee across body to the left','Extend right arm out, look right','Left hand rests on right knee','Hold 3 min, repeat other side']),
  ]),
  WorkoutDay(day: 'Saturday', focus: 'Light Cardio + Breathwork', exercises: [
    _ex('Cycling (Easy Pace)','20–25 min','Low','Gentle aerobic boost without pelvic jarring',Icons.pedal_bike,['Set resistance to easy level','Pedal at comfortable 50–60 RPM','Keep back straight, avoid leaning hard forward','Breathe evenly throughout','Cool down last 5 min at slower pace']),
    _ex('4-7-8 Breathing','10 min','Low','Activates relaxation response, reduces chronic pain',Icons.air,['Sit comfortably, spine tall','Inhale through nose for 4 counts','Hold breath for 7 counts','Exhale through mouth fully for 8 counts','Repeat 8 cycles twice daily']),
    _ex('Shoulder & Neck Rolls','5 min','Low','Releases upper body tension held during pain episodes',Icons.self_improvement,['Sit comfortably or stand','Slowly roll neck in full circles (3 each way)','Roll shoulders forward 5 times, backward 5 times','Tilt ear to shoulder, hold 10 sec each side','Finish with 5 deep breaths']),
  ]),
  WorkoutDay(day: 'Sunday', focus: 'Full Rest & Meditation', exercises: [
    _ex('Complete Rest Day','All day','Low','Essential recovery reduces inflammation markers',Icons.nights_stay,['No structured exercise today','Gentle stretching only if desired','Sleep 8+ hours to support repair','Apply heat pad to pelvic area if needed','Nourish with anti-inflammatory foods & fluids']),
    _ex('Body Scan Meditation','15 min','Low','Reduces pain perception & anxiety',Icons.psychology,['Lie comfortably, eyes closed','Bring attention to toes — notice sensation','Slowly scan upward through entire body','Release tension in each area as you pass','End visualising warm healing light']),
    _ex('Savasana','10 min','Low','Deep nervous system rest & restoration',Icons.self_improvement,['Lie flat, legs slightly apart, palms up','Close eyes and release all tension','Breathe naturally with no control','If mind wanders, gently return to breath','Stay completely still for 10 full minutes']),
  ]),
];

// ─────────────────────────────────────────────
//  THYROID DATA (7 days)
// ─────────────────────────────────────────────

final List<DayPlan> _thyroidMeals = [
  DayPlan(day: 'Monday',
    breakfast: _m('Thyroid Power Eggs','Selenium & iodine breakfast','310 kcal',['2 eggs (scrambled)','1 nori sheet (crumbled on top)','Sautéed mushrooms','1 slice GF toast','3 Brazil nuts','Herbal tea']),
    lunch: _m('Chicken & Sweet Potato','Energy-boosting balanced lunch','470 kcal',['150g grilled chicken','1 medium baked sweet potato','Steamed broccoli (cooked)','Olive oil & rosemary','Mixed greens salad']),
    dinner: _m('Miso Soup & Rice','Iodine-rich Japanese dinner','440 kcal',['1 bowl miso soup (seaweed, tofu, scallion)','½ cup brown rice','100g steamed fish','Pickled ginger','Edamame (small portion)']),
    snack: _m('Energy Snack','Thyroid-boosting snack','170 kcal',['1 boiled egg','1 small orange','3 Brazil nuts']),
  ),
  DayPlan(day: 'Tuesday',
    breakfast: _m('Berry Protein Smoothie','Antioxidant morning smoothie','300 kcal',['1 cup blueberries','1 banana','1 scoop vanilla protein','1 tbsp flaxseeds','Almond milk','1 tsp ashwagandha powder']),
    lunch: _m('Tuna & Quinoa Bowl','Selenium-rich lunch bowl','480 kcal',['1 can tuna (in water)','½ cup quinoa','Cherry tomatoes & cucumber','Capers & lemon juice','1 tbsp olive oil']),
    dinner: _m('Lamb & Roasted Veg','Zinc-rich dinner','510 kcal',['150g lean lamb','Roasted zucchini & bell peppers','½ cup millet','Garlic & thyme','Side salad']),
    snack: _m('Iodine Boost Snack','Sea-vegetable snack','100 kcal',['1 pack roasted seaweed snacks','5 almonds','Herbal tea']),
  ),
  DayPlan(day: 'Wednesday',
    breakfast: _m('Oat & Seed Bowl','Iron-boosting breakfast','320 kcal',['½ cup oats','1 tbsp pumpkin seeds','1 tbsp hemp seeds','Sliced banana','Fortified almond milk','Cinnamon']),
    lunch: _m('Salmon & Lentil Salad','Complete thyroid-support meal','490 kcal',['130g baked salmon','½ cup puy lentils','Roasted beets','Arugula','Balsamic dressing']),
    dinner: _m('Turkey & Veggie Stir-Fry','B12-rich lean protein dinner','460 kcal',['150g ground turkey','Bok choy, snap peas, carrots','Tamari sauce (low sodium)','½ cup rice noodles','Sesame seeds']),
    snack: _m('Vitamin D Snack','Bone & thyroid support','150 kcal',['1 cup fortified coconut yogurt','1 tbsp sunflower seeds','Honey drizzle']),
  ),
  DayPlan(day: 'Thursday',
    breakfast: _m('Egg & Seaweed Wrap','Iodine-packed morning wrap','300 kcal',['2 scrambled eggs','1 nori wrap','Sliced avocado','Cucumber & sesame seeds','Tamari dipping sauce']),
    lunch: _m('Chicken & Buckwheat Bowl','Zinc & B-vitamin rich bowl','460 kcal',['140g grilled chicken','½ cup buckwheat','Roasted red pepper & courgette','Tahini dressing','Mixed herbs']),
    dinner: _m('Prawn & Vegetable Curry','Iodine-rich seafood dinner','470 kcal',['150g prawns','Coconut milk curry base','Cooked spinach & tomato','½ cup brown rice','Fresh coriander']),
    snack: _m('Selenium Snack','Thyroid mineral boost','120 kcal',['3 Brazil nuts','½ cup sunflower seeds','1 kiwi fruit']),
  ),
  DayPlan(day: 'Friday',
    breakfast: _m('Ashwagandha Smoothie Bowl','Adaptogen-rich energising bowl','330 kcal',['1 frozen banana','1 cup mixed berries','1 tsp ashwagandha powder','1 tbsp hemp seeds','Granola topping','Almond milk base']),
    lunch: _m('Sardine Toast & Salad','Omega-3 & iodine quick lunch','440 kcal',['2 GF crackers','1 can sardines in olive oil','Sliced tomato','Rocket leaves & lemon','Olive oil drizzle']),
    dinner: _m('Beef & Root Vegetable Stew','Iron & zinc-rich slow meal','520 kcal',['130g lean beef','Carrots, parsnip & sweet potato','Onion, garlic & thyme','Low-sodium beef broth','Side of cooked kale']),
    snack: _m('B12 Boost Snack','B12 & protein snack','140 kcal',['1 boiled egg','½ cup fortified oat milk','5 walnuts']),
  ),
  DayPlan(day: 'Saturday',
    breakfast: _m('Coconut Chia Pudding','Easy anti-inflammatory breakfast','290 kcal',['3 tbsp chia seeds soaked overnight','Coconut milk','Sliced mango & kiwi','1 tsp vanilla extract','Toasted coconut flakes']),
    lunch: _m('Nicoise-Style Salad','Iodine & iron loaded lunch','470 kcal',['1 can tuna in water','Boiled egg','Green beans, olives, capers','Lettuce & cherry tomatoes','Lemon-olive oil dressing']),
    dinner: _m('Baked Cod with Quinoa','Selenium & lean protein dinner','460 kcal',['160g baked cod','½ cup quinoa','Steamed asparagus & courgette','Lemon & dill sauce','Side green salad']),
    snack: _m('Zinc-Rich Trail Mix','Afternoon energy blend','160 kcal',['1 tbsp pumpkin seeds','1 tbsp cashews','5 almonds','2 dried apricots']),
  ),
  DayPlan(day: 'Sunday',
    breakfast: _m('Warm Porridge & Berries','Comforting iron-rich breakfast','320 kcal',['½ cup gluten-free oats','1 tbsp hemp seeds','Stewed plums or prunes','1 tsp blackstrap molasses','Oat milk']),
    lunch: _m('Chicken & Vegetable Soup','Comforting nourishing soup','430 kcal',['150g shredded chicken','Carrot, celery & onion','Low-sodium chicken broth','½ cup rice noodles','Fresh parsley']),
    dinner: _m('Mushroom & Egg Fried Rice','Selenium & B12 loaded dinner','450 kcal',['1 cup brown rice','2 eggs scrambled in','Shiitake & oyster mushrooms','Tamari sauce & garlic','Spring onions & sesame oil']),
    snack: _m('Evening Wind-Down Snack','Magnesium calming snack','120 kcal',['1 cup chamomile or ashwagandha tea','3 Brazil nuts','½ banana']),
  ),
];

final List<WorkoutDay> _thyroidWorkout = [
  WorkoutDay(day: 'Monday', focus: 'Moderate Cardio', exercises: [
    _ex('Brisk Walking / Light Jog','30–40 min','Moderate','Boosts sluggish metabolism & lifts energy levels',Icons.directions_walk,['Warm up with 5 min slow walk','Increase to brisk pace (can hold conversation)','Swing arms to increase calorie burn','Focus on even, steady breathing','Cool down with 5 min slow walk & stretch']),
    _ex('Cycling (Stationary or Outdoor)','25 min','Moderate','Cardiovascular boost without joint stress',Icons.pedal_bike,['Set bike resistance to moderate level','Maintain a cadence of 60–80 RPM','Keep back straight, core engaged','Breathe steadily throughout','Cool down last 5 min at easy pace']),
    _ex('Neck Rolls & Shoulder Stretches','5 min','Low','Improves circulation to thyroid gland in neck',Icons.self_improvement,['Sit comfortably, spine tall','Gently roll neck in slow circles (3 each direction)','Tilt ear to shoulder, hold 10 sec each side','Roll shoulders forward and backward 5 times','Finish with chin tucks (5 reps)']),
  ]),
  WorkoutDay(day: 'Tuesday', focus: 'Yoga — Thyroid Poses', exercises: [
    _ex('Shoulder Stand (Sarvangasana)','2–3 min hold','Moderate','Directly stimulates & tones the thyroid gland',Icons.self_improvement,['Lie on back, arms by sides','Lift legs and hips off ground using core','Support hips with hands, elbows on mat','Straighten body toward the ceiling','Hold, breathe steadily — come down slowly']),
    _ex('Plow Pose (Halasana)','1–2 min hold','Moderate','Stretches neck & activates thyroid area',Icons.self_improvement,['From shoulder stand, lower legs behind head','Toes touch the floor behind you','Keep arms flat on mat or clasp hands','Chin presses gently toward chest','Hold 1–2 min, come out slowly']),
    _ex('Fish Pose (Matsyasana)','2 min hold','Low','Opens throat chakra, stimulates thyroid',Icons.self_improvement,['Lie on back, slide hands under hips','Press elbows down and arch chest upward','Tilt head back so crown rests on mat','Breathe deeply into the chest','Hold 2 min, release gently']),
  ]),
  WorkoutDay(day: 'Wednesday', focus: 'Strength Training', exercises: [
    _ex('Resistance Band Rows','3 × 12 reps','Moderate','Builds muscle to raise resting metabolic rate',Icons.fitness_center,['Anchor band at chest height','Hold handles, step back for tension','Pull handles toward chest, elbows back','Squeeze shoulder blades together at end','Slowly return to start position']),
    _ex('Step-Ups','3 × 12 each leg','Moderate','Builds leg strength & boosts metabolism',Icons.fitness_center,['Stand facing a step or sturdy chair','Step right foot up, bring left to meet it','Step back down, right foot first','Alternate leading leg each set','Hold light weights to increase intensity']),
    _ex('Wall Sit','3 × 40 sec','Moderate','Activates large muscle groups, burns more calories',Icons.fitness_center,['Stand with back against wall','Slide down until thighs are parallel to floor','Keep back flat against wall','Hold position, breathe steadily','Rest 30 sec between holds']),
  ]),
  WorkoutDay(day: 'Thursday', focus: 'Swimming / Water Exercise', exercises: [
    _ex('Lap Swimming','25–30 min','Moderate','Full-body metabolism boost without overexertion',Icons.pool,['Warm up with 2 min easy freestyle','Swim at comfortable moderate pace','Mix freestyle and breaststroke','Focus on rhythmic, controlled breathing','Cool down with gentle backstroke']),
    _ex('Aqua Jogging','15 min','Moderate','Cardio with zero impact on joints',Icons.pool,['Stand in chest-deep water','Jog with knees high, arms pumping','Keep core tight throughout','Maintain upright posture','Progress from 10 to 20 min over weeks']),
    _ex('Seated Spinal Twist','3 min each side','Low','Stimulates thyroid & improves energy flow',Icons.self_improvement,['Sit cross-legged or in a chair','Place right hand on left knee','Left hand behind on floor or chair','Inhale to lengthen spine','Exhale and twist left, hold 3 min, repeat other side']),
  ]),
  WorkoutDay(day: 'Friday', focus: 'Gentle HIIT', exercises: [
    _ex('Jumping Jacks','3 × 45 sec','Moderate','Full-body cardio, elevates heart rate gently',Icons.directions_run,['Start with feet together, arms by sides','Jump feet out while raising arms overhead','Jump back to starting position','Keep landing soft, knees slightly bent','Maintain a steady rhythm']),
    _ex('Bodyweight Squats','3 × 15 reps','Moderate','Activates large muscle groups, boosts metabolism',Icons.fitness_center,['Stand feet shoulder-width apart','Lower into squat keeping chest up','Descend until thighs are parallel','Drive through heels to stand','Squeeze glutes at top']),
    _ex('Marching in Place','3 × 1 min','Low','Gentle cardio warm-up, combats fatigue',Icons.directions_walk,['Stand tall and march on the spot','Bring knees up to hip height','Swing arms naturally','Breathe rhythmically','Increase pace gradually']),
  ]),
  WorkoutDay(day: 'Saturday', focus: 'Active Leisure', exercises: [
    _ex('Nature Hike or Long Walk','45–60 min','Moderate','Sunshine + movement raises Vitamin D & serotonin',Icons.landscape,['Choose a scenic route or park','Maintain comfortable walking pace','Breathe fresh air deeply','Take breaks as needed','Stretch legs & back after']),
    _ex('Tai Chi (Basic Form)','20 min','Low','Reduces cortisol, improves energy flow',Icons.self_improvement,['Stand with feet shoulder-width apart','Move arms slowly through flowing positions','Focus on breath with each movement','Shift weight gently side to side','Follow a beginner video if needed']),
    _ex('Gentle Stretching Routine','10 min','Low','Improves flexibility & combats stiffness',Icons.self_improvement,['Stretch hamstrings, quads & hip flexors','Hold each stretch 30 seconds','Breathe deeply into each position','Never force or bounce','Finish with chest and shoulder opening stretch']),
  ]),
  WorkoutDay(day: 'Sunday', focus: 'Rest & Restorative Yoga', exercises: [
    _ex('Restorative Rest Day','All day','Low','Recovery & repair essential for thyroid healing',Icons.nights_stay,['Prioritise 8 hours of quality sleep','Avoid screens 1 hour before bed','Practice gentle breathing exercises','Hydrate well with herbal teas','Prepare healthy meals for the coming week']),
    _ex('Legs Up the Wall','10 min','Low','Reduces leg fatigue, calms nervous system',Icons.self_improvement,['Lie on back near a wall','Swing legs up against the wall','Arms relaxed by sides, palms up','Breathe deeply for 10 minutes','Come out slowly by rolling to one side']),
    _ex('Guided Meditation','15 min','Low','Reduces cortisol that suppresses thyroid function',Icons.psychology,['Sit or lie comfortably','Close eyes and follow your breath','Use a guided app or simply observe thoughts','Let go of any physical or mental tension','End with 3 slow deep breaths']),
  ]),
];

// ─────────────────────────────────────────────
//  CERVICAL CANCER DATA (7 days)
// ─────────────────────────────────────────────

final List<DayPlan> _cervicalMeals = [
  DayPlan(day: 'Monday',
    breakfast: _m('Folate Power Bowl','Folate & antioxidant-rich start','310 kcal',['½ cup spinach (sautéed)','2 whole eggs (poached)','1 slice whole-grain toast','½ cup sliced strawberries','Green tea or warm lemon water']),
    lunch: _m('Lycopene Salad Bowl','Tomato & greens protective bowl','430 kcal',['1 cup cherry tomatoes (roasted)','2 cups mixed dark greens','½ cup quinoa','½ avocado','2 tbsp olive oil & lemon dressing','Pumpkin seeds']),
    dinner: _m('Turmeric Lentil Dal','Immune-boosting anti-inflammatory dinner','470 kcal',['1 cup red lentils','Turmeric, garlic & ginger','1 cup sautéed kale','½ cup brown rice','Fresh coriander & lemon']),
    snack: _m('Vitamin C Boost','Citrus & antioxidant snack','120 kcal',['1 medium orange','½ cup papaya cubes','5 almonds','Warm green tea']),
  ),
  DayPlan(day: 'Tuesday',
    breakfast: _m('Berry Chia Pudding','Antioxidant overnight pudding','290 kcal',['3 tbsp chia seeds','1 cup unsweetened almond milk','½ cup mixed berries','1 tsp flaxseeds','1 tsp raw honey']),
    lunch: _m('Cruciferous Stir-Fry','Indole-rich cancer-protective lunch','440 kcal',['1 cup broccoli & cauliflower florets','½ cup cabbage (shredded)','150g tofu or grilled chicken','Garlic, ginger & tamari sauce','½ cup brown rice']),
    dinner: _m('Baked Salmon & Greens','Selenium & omega-3 protective dinner','490 kcal',['150g baked salmon','1 cup steamed spinach & asparagus','½ cup lentils','Lemon-dill dressing','Cherry tomatoes']),
    snack: _m('Beta-Carotene Bites','Carrot & nut protective snack','130 kcal',['Carrot & cucumber sticks','2 tbsp hummus','1 tsp sesame seeds','Herbal turmeric tea']),
  ),
  DayPlan(day: 'Wednesday',
    breakfast: _m('Green Immunity Smoothie','Folate & Vitamin C morning blend','280 kcal',['1 cup spinach','½ cup mango chunks','½ cup pineapple','1 tbsp flaxseeds','1 cup coconut water','1 tsp spirulina (optional)']),
    lunch: _m('Mediterranean Chickpea Plate','Folate & antioxidant-rich plate','450 kcal',['¾ cup roasted chickpeas','Roasted red peppers & zucchini','½ cup farro or quinoa','2 tbsp hummus','Olive oil, lemon & parsley']),
    dinner: _m('Garlic Chicken & Sweet Potato','Immune-strengthening dinner','480 kcal',['150g grilled garlic chicken breast','1 medium baked sweet potato','1 cup steamed broccoli','Olive oil & rosemary','Side green salad']),
    snack: _m('Lycopene Evening Snack','Tomato & seed protective snack','110 kcal',['1 cup tomato juice (no added salt)','1 tbsp sunflower seeds','5 walnut halves']),
  ),
  DayPlan(day: 'Thursday',
    breakfast: _m('Papaya Folate Bowl','Enzyme & folate rich breakfast','270 kcal',['1 cup papaya chunks','½ cup Greek yogurt (dairy-free)','1 tbsp hemp seeds','Lime zest','Warm green tea']),
    lunch: _m('Warm Lentil & Kale Soup','Folate & iron immunity soup','420 kcal',['1 cup green lentils','2 cups kale','Diced tomatoes & garlic','Low-sodium vegetable broth','Lemon juice & cumin']),
    dinner: _m('Turmeric Prawn Stir-Fry','Selenium & anti-inflammatory dinner','460 kcal',['150g prawns','Turmeric & garlic sauce','Bok choy, snap peas & carrots','½ cup brown rice','Fresh coriander']),
    snack: _m('Immune Boost Snack','Vitamin C & E combo snack','130 kcal',['1 kiwi','½ cup strawberries','1 tbsp sunflower seeds','Green tea']),
  ),
  DayPlan(day: 'Friday',
    breakfast: _m('Avocado & Tomato Toast','Lycopene & healthy fats breakfast','320 kcal',['2 slices whole-grain bread','½ avocado mashed','Sliced tomatoes','Fresh basil & olive oil','Lemon squeeze']),
    lunch: _m('Rainbow Buddha Bowl','Full-spectrum antioxidant bowl','460 kcal',['½ cup quinoa','Roasted sweet potato & beets','Shredded red cabbage','½ avocado','Tahini lemon dressing']),
    dinner: _m('Herb Baked Cod & Lentils','Selenium & folate rich dinner','470 kcal',['160g baked cod','Fresh dill & lemon herb crust','½ cup green lentils','Steamed asparagus','Cherry tomato salsa']),
    snack: _m('Antioxidant Berry Cup','Free-radical fighting snack','100 kcal',['½ cup blueberries','½ cup pomegranate seeds','1 tsp chia seeds','Warm green tea']),
  ),
  DayPlan(day: 'Saturday',
    breakfast: _m('Spinach Omelette','Folate-packed weekend breakfast','300 kcal',['3 egg whites + 1 yolk','1 cup spinach','Diced tomato & red onion','Fresh herbs & black pepper','Whole-grain toast']),
    lunch: _m('Grilled Veggie & Hummus Plate','Antioxidant-rich colorful plate','420 kcal',['Grilled courgette, aubergine & peppers','½ cup hummus','1 whole-wheat pita','Fresh parsley & lemon','Cherry tomatoes']),
    dinner: _m('Sweet Potato & Black Bean Curry','Beta-carotene & folate dinner','480 kcal',['1 medium sweet potato','½ cup black beans','Coconut milk curry base','Garlic, ginger & turmeric','½ cup brown rice']),
    snack: _m('Walnut & Citrus Snack','Vitamin E & C combo','130 kcal',['6 walnuts','1 small clementine','1 tsp flaxseeds','Chamomile tea']),
  ),
  DayPlan(day: 'Sunday',
    breakfast: _m('Tropical Immunity Smoothie','Vitamin C & folate power blend','280 kcal',['½ cup mango','½ cup pineapple','1 cup spinach','1 tbsp flaxseeds','1 cup coconut water','1 tsp turmeric']),
    lunch: _m('Roasted Tomato & Lentil Soup','Lycopene-rich comforting soup','410 kcal',['4 roasted tomatoes','½ cup red lentils','Garlic & onion base','Low-sodium vegetable broth','Fresh basil & olive oil']),
    dinner: _m('Baked Chicken & Broccoli Bake','Indole & lean protein dinner','470 kcal',['160g chicken breast','2 cups broccoli florets','Garlic, lemon & olive oil','Cherry tomatoes & herbs','Side of quinoa']),
    snack: _m('Healing Sunday Snack','Calming immune-supportive snack','110 kcal',['1 cup green tea with lemon','5 almonds','½ cup raspberries']),
  ),
];

final List<WorkoutDay> _cervicalWorkout = [
  WorkoutDay(day: 'Monday', focus: 'Immune Walk & Breathwork', exercises: [
    _ex('Nature Walk','30–40 min','Low','Boosts NK cell activity, supports immunity',Icons.directions_walk,['Choose a green space or park if possible','Walk at a comfortable, energizing pace','Breathe fresh air deeply with each step','Swing arms naturally for balance','End with gentle stretching of legs & back']),
    _ex('4-7-8 Breathing','10 min','Low','Reduces cortisol, activates rest-digest system',Icons.air,['Sit comfortably with spine tall','Inhale through nose for 4 counts','Hold breath for 7 counts','Exhale fully through mouth for 8 counts','Repeat 8 cycles, twice daily']),
    _ex('Sun Salutation (Surya Namaskar)','5 rounds','Low','Full-body gentle activation, reduces fatigue',Icons.self_improvement,['Start standing, hands at heart center','Inhale, raise arms overhead','Exhale, forward fold','Step back to plank, lower down slowly','Upward dog then downward dog — hold 5 breaths']),
  ]),
  WorkoutDay(day: 'Tuesday', focus: 'Gentle Strength', exercises: [
    _ex('Wall Push-Ups','3 × 12 reps','Low','Builds upper body strength gently without strain',Icons.fitness_center,['Stand arm\'s length from wall','Place palms flat on wall at shoulder height','Bend elbows and lean chest toward wall','Push back to starting position','Keep core tight throughout']),
    _ex('Chair Squats','3 × 10 reps','Low','Leg strength without over-exerting',Icons.fitness_center,['Stand in front of a chair','Lower slowly as if sitting down','Just touch the chair, then stand back up','Keep chest up, knees over toes','Use arms for balance if needed']),
    _ex('Seated Arm Raises','3 × 12 reps','Low','Maintains upper body strength during recovery',Icons.fitness_center,['Sit on chair with light weights (or water bottles)','Raise both arms forward to shoulder height','Lower slowly with control','Keep core lightly engaged','Breathe out on the raise, in on the lower']),
  ]),
  WorkoutDay(day: 'Wednesday', focus: 'Pelvic Health & Yoga', exercises: [
    _ex('Kegel Exercises','3 × 15 reps','Low','Strengthens pelvic floor, supports recovery',Icons.fitness_center,['Lie down or sit comfortably','Identify pelvic floor muscles','Contract and hold for 5 seconds','Release slowly for 5 seconds','Repeat 15 times, 3 times daily']),
    _ex('Chair Yoga — Seated Twist','5 min each side','Low','Stimulates lymphatic drainage, gentle detox',Icons.self_improvement,['Sit on edge of chair, feet flat','Inhale and lengthen spine upward','Exhale, twist right — right hand on chair back, left on right knee','Hold 5 breaths, gaze over right shoulder','Return to centre, repeat left side']),
    _ex('Supported Child\'s Pose','5 min','Low','Gently stretches lower back, pelvic area',Icons.self_improvement,['Place a bolster or pillow lengthwise on mat','Kneel and fold forward over the support','Rest head and arms on the bolster','Breathe deeply, let go of tension','Stay 5 minutes with eyes closed']),
  ]),
  WorkoutDay(day: 'Thursday', focus: 'Cardio Walk & Stretching', exercises: [
    _ex('Interval Walking','30 min','Low','Gentle cardio intervals boost immunity',Icons.directions_walk,['Walk at normal pace 2 min','Walk briskly for 1 min','Alternate for 30 minutes total','Focus on steady breathing','Cool down with 5 min gentle pace']),
    _ex('Standing Hip Circles','3 min','Low','Mobilises pelvis, improves pelvic circulation',Icons.self_improvement,['Stand feet hip-width apart','Hands on hips','Draw large slow circles with hips','10 clockwise, 10 counter-clockwise','Move slowly and breathe deeply']),
    _ex('Quad & Hamstring Stretch','3 min each leg','Low','Releases leg tension, improves lymph flow',Icons.self_improvement,['Hold wall for balance','Bend right knee, hold ankle behind you','Stand tall, feel stretch in front of thigh','Hold 30 sec, release','Then reach for toes for hamstring stretch']),
  ]),
  WorkoutDay(day: 'Friday', focus: 'Restorative Yoga', exercises: [
    _ex('Supported Bridge Pose','5 min hold','Low','Opens chest, reduces fatigue, supports hormonal balance',Icons.self_improvement,['Lie on back, knees bent, feet flat','Place yoga block or blanket under sacrum','Let hips rest supported on the block','Arms relaxed by sides, palms up','Close eyes and breathe deeply for 5 minutes']),
    _ex('Reclining Butterfly Pose','5 min','Low','Gently opens hips, calms pelvic area',Icons.self_improvement,['Lie on back','Bring soles of feet together, knees fall open','Place hands on belly or by sides','Breathe deeply into lower belly','Use cushions under knees for support']),
    _ex('Savasana','10 min','Low','Deep nervous system restoration, reduces fatigue',Icons.self_improvement,['Lie flat on back, legs slightly apart','Arms at sides, palms facing up','Close eyes and release all tension','Breathe naturally, no control needed','Stay completely still for 10 minutes']),
  ]),
  WorkoutDay(day: 'Saturday', focus: 'Swimming & Meditation', exercises: [
    _ex('Gentle Swimming','20–25 min','Low','Whole-body movement without pelvic strain',Icons.pool,['Choose breaststroke or backstroke','Swim at easy, comfortable pace','Focus on long, smooth strokes','Breathe rhythmically','Gentle floating stretch to finish']),
    _ex('Body Scan Meditation','15 min','Low','Reduces anxiety & supports healing mindset',Icons.psychology,['Lie comfortably, eyes closed','Bring attention to toes — notice sensation','Slowly scan upward through entire body','Release tension in each area as you pass','End visualising warm healing light through body']),
    _ex('Wrist & Ankle Circles','5 min','Low','Stimulates lymphatic drainage in extremities',Icons.self_improvement,['Sit comfortably','Slowly rotate each wrist 10 times each direction','Then rotate each ankle 10 times each direction','Flex and point feet 10 times','Shake hands and feet loosely to finish']),
  ]),
  WorkoutDay(day: 'Sunday', focus: 'Full Rest & Healing', exercises: [
    _ex('Complete Rest Day','All day','Low','Rest is essential for immune repair & cell recovery',Icons.nights_stay,['No structured exercise today','Short gentle walk if energy allows','Prioritise 8–9 hours of sleep','Practice deep breathing or prayer','Nourish with anti-inflammatory foods & warm fluids']),
    _ex('Gratitude Journaling','15 min','Low','Positive mindset improves immune function',Icons.edit_note,['Find a quiet, comfortable space','Write 3 things you are grateful for today','Note any progress in your wellbeing','Set a gentle intention for the coming week','Finish with 5 slow, deep breaths']),
    _ex('Progressive Muscle Relaxation','10 min','Low','Releases physical tension, improves sleep quality',Icons.self_improvement,['Lie down in savasana position','Tense each muscle group for 5 seconds','Release fully and notice the difference','Work from toes to head systematically','End with 5 deep belly breaths']),
  ]),
];

// ─────────────────────────────────────────────
//  ASSEMBLED PLANS
// ─────────────────────────────────────────────

final List<DiseaseDietPlan> dietPlans = [
  DiseaseDietPlan(
    disease: 'PCOS / PCOD', subtitle: 'Polycystic Ovarian Syndrome',
    description: 'A low-GI, anti-inflammatory diet helps regulate insulin levels and balance hormones naturally.',
    icon: Icons.spa_outlined, primaryColor: const Color(0xFFC85A7A), accentColor: const Color(0xFFFFE4EC),
    keyNutrients: ['Omega-3 Fatty Acids','Magnesium','Zinc','Inositol (B8)','Chromium','Vitamin D'],
    foodsToAvoid: ['Refined sugar & white flour','Processed & fried foods','Dairy (limit)','Alcohol','High-GI fruits (mango, grapes)','Soy products'],
    superfoods: ['🥦 Broccoli','🫐 Blueberries','🌿 Spearmint tea','🐟 Salmon','🥑 Avocado','🌰 Walnuts','🫘 Lentils','🍃 Flaxseeds'],
    weeklyPlan: _pcosMeals, workoutWeek: _pcosWorkout,
    exerciseOverview: 'For PCOS, a mix of strength training and low-impact cardio improves insulin sensitivity, reduces androgens and supports healthy weight management. Avoid over-exercising — it can spike cortisol and worsen symptoms.',
    exerciseTips: ['Exercise 4–5 days/week, 30–45 min sessions','Combine strength training with walking or yoga','Avoid prolonged high-intensity cardio daily','Morning exercise helps regulate blood sugar','Rest adequately — overtraining worsens PCOS'],
  ),
  DiseaseDietPlan(
    disease: 'Endometriosis', subtitle: 'Endometrial Tissue Disorder',
    description: 'An anti-inflammatory, estrogen-reducing diet helps manage pain and slow abnormal tissue growth.',
    icon: Icons.favorite_border, primaryColor: const Color(0xFF7C4D9F), accentColor: const Color(0xFFF3E8FF),
    keyNutrients: ['Omega-3 Fatty Acids','Antioxidants','Vitamin E','Iron','B Vitamins','Magnesium'],
    foodsToAvoid: ['Red meat & processed meat','Trans fats','Caffeine (limit)','Gluten (some benefit)','Alcohol','High-estrogen foods'],
    superfoods: ['🫐 Berries (all types)','🥬 Dark leafy greens','🐟 Fatty fish','🫚 Olive oil','🍵 Green tea','🧄 Garlic & turmeric','🥕 Carrots & beets','🌰 Brazil nuts'],
    weeklyPlan: _endoMeals, workoutWeek: _endoWorkout,
    exerciseOverview: 'Gentle, low-impact exercise reduces inflammation, balances estrogen and relieves pelvic pain. During flare-ups, prioritize restorative yoga and walking. Avoid high-impact activities that worsen pain.',
    exerciseTips: ['Listen to your body — rest during flare-ups','Yoga & Pilates are ideal for pain relief','Gentle swimming reduces pelvic pressure','Avoid high-impact workouts during periods','Heat + gentle stretching relieves cramps'],
  ),
  DiseaseDietPlan(
    disease: 'Thyroid Disorders', subtitle: 'Hypothyroidism & Hashimoto\'s',
    description: 'A thyroid-supporting diet rich in iodine, selenium and zinc helps optimize thyroid hormone production.',
    icon: Icons.self_improvement, primaryColor: const Color(0xFF2E86AB), accentColor: const Color(0xFFE0F4FF),
    keyNutrients: ['Iodine','Selenium','Zinc','Vitamin D','Iron','B12'],
    foodsToAvoid: ['Raw cruciferous veggies (large amounts)','Soy products','Gluten (Hashimoto\'s)','Highly processed foods','Excess sugar','Fluoride-heavy water'],
    superfoods: ['🐟 Seaweed & kelp','🥚 Eggs','🫘 Brazil nuts (selenium)','🍗 Lean chicken','🫐 Berries','🍠 Sweet potato','🌱 Ashwagandha','🧅 Cooked cruciferous veg'],
    weeklyPlan: _thyroidMeals, workoutWeek: _thyroidWorkout,
    exerciseOverview: 'Hypothyroidism causes fatigue and slow metabolism. Regular moderate exercise boosts metabolism, energy levels and mood. Avoid extreme overexertion which may worsen thyroid autoimmune response.',
    exerciseTips: ['Start slow — even 20 min/day makes a difference','Morning workouts help combat fatigue','Swimming is excellent for thyroid patients','Yoga reduces cortisol that suppresses thyroid','Consistency matters more than intensity'],
  ),
  DiseaseDietPlan(
    disease: 'Cervical Cancer', subtitle: 'Nutritional Support & Prevention',
    description: 'A diet rich in antioxidants, folate and immune-boosting nutrients helps protect cervical cells, reduce HPV progression risk and support recovery.',
    icon: Icons.shield_outlined, primaryColor: const Color(0xFF1E8A6E), accentColor: const Color(0xFFDFF5EF),
    keyNutrients: ['Folate (B9)','Vitamin C','Vitamin E','Beta-Carotene','Lycopene','Selenium'],
    foodsToAvoid: ['Processed & packaged meats','Alcohol','Refined carbs & white sugar','Deep-fried foods','Artificial additives & preservatives','High-sodium canned foods'],
    superfoods: ['🍅 Tomatoes (lycopene)','🥦 Broccoli & cruciferous veg','🫐 Berries (antioxidants)','🥬 Dark leafy greens (folate)','🧄 Garlic (immune support)','🍋 Citrus fruits (Vitamin C)','🥕 Carrots & sweet potato','🌿 Turmeric & green tea'],
    weeklyPlan: _cervicalMeals, workoutWeek: _cervicalWorkout,
    exerciseOverview: 'Regular moderate exercise strengthens the immune system, reduces inflammation, supports healthy cell function and improves quality of life during and after cervical cancer treatment. Always consult your doctor before starting.',
    exerciseTips: ['Consult your oncologist before starting any program','Moderate exercise strengthens immunity & reduces recurrence risk','Walking daily is safe even during treatment','Avoid overexertion — listen to your body','Yoga & breathing reduce treatment-related fatigue'],
  ),
];

// ─────────────────────────────────────────────
//  PAGE & UI
// ─────────────────────────────────────────────

class DietPlanPage extends StatefulWidget {
  const DietPlanPage({Key? key}) : super(key: key);
  @override
  State<DietPlanPage> createState() => _DietPlanPageState();
}

class _DietPlanPageState extends State<DietPlanPage> with TickerProviderStateMixin {
  String dietPlan = '';
  bool isLoading = true;
  int _selectedDiseaseIndex = 0;
  int _selectedDayIndex = 0;
  int _selectedWorkoutDayIndex = 0;
  late TabController _tabController;

  Color _intensityColor(String intensity) {
    switch (intensity) {
      case 'High':     return const Color(0xFFE53E3E);
      case 'Moderate': return const Color(0xFFD97706);
      default:         return const Color(0xFF2D9E6B);
    }
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _tabController.addListener(() {
      if (!_tabController.indexIsChanging) {
        setState(() {
          _selectedDiseaseIndex = _tabController.index;
          _selectedDayIndex = 0;
          _selectedWorkoutDayIndex = 0;
        });
      }
    });
    fetchDietPlan();
  }

  Future<void> fetchDietPlan() async {
    setState(() => isLoading = true);
    final result = await GroqService().generateDietPlan('irregular periods, hormonal imbalance');
    setState(() { dietPlan = result; isLoading = false; });
  }

  @override
  void dispose() { _tabController.dispose(); super.dispose(); }

  DiseaseDietPlan get _currentPlan => dietPlans[_selectedDiseaseIndex];
  DayPlan get _currentDay => _currentPlan.weeklyPlan[_selectedDayIndex];
  WorkoutDay get _currentWorkout => _currentPlan.workoutWeek[_selectedWorkoutDayIndex];

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
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [_currentPlan.primaryColor, _currentPlan.primaryColor.withOpacity(0.7)]),
      ),
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Row(
        children: [
          IconButton(onPressed: () => Navigator.pop(context), icon: const Icon(Icons.arrow_back, color: Colors.white, size: 26)),
          const SizedBox(width: 8),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Diet & Exercise Guide', style: TextStyle(color: Colors.white, fontSize: 21, fontWeight: FontWeight.w800)),
            Text('Personalized wellness for women\'s health', style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 12)),
          ]),
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
        tabs: dietPlans.map((plan) => Tab(child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Text(plan.disease)))).toList(),
      ),
    );
  }

  Widget _buildDietOverviewCard() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [_currentPlan.primaryColor, _currentPlan.primaryColor.withOpacity(0.75)]),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: _currentPlan.primaryColor.withOpacity(0.3), blurRadius: 15, offset: const Offset(0, 5))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Container(padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: Colors.white.withOpacity(0.25), borderRadius: BorderRadius.circular(12)),
              child: Icon(_currentPlan.icon, color: Colors.white, size: 28)),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(_currentPlan.disease, style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
            Text(_currentPlan.subtitle, style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 12)),
          ])),
        ]),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(12)),
          child: Text(_currentPlan.description, style: const TextStyle(color: Colors.white, fontSize: 13.5, height: 1.5)),
        ),
      ]),
    );
  }

  Widget _buildSuperFoodsSection() {
    return _buildSection(title: '⭐ Superfoods to Include', color: _currentPlan.primaryColor,
      child: Wrap(spacing: 8, runSpacing: 8,
        children: _currentPlan.superfoods.map((food) => Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(color: _currentPlan.accentColor, borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _currentPlan.primaryColor.withOpacity(0.3))),
          child: Text(food, style: TextStyle(color: _currentPlan.primaryColor, fontWeight: FontWeight.w600, fontSize: 13)),
        )).toList(),
      ),
    );
  }

  Widget _buildFoodsToAvoidSection() {
    return _buildSection(title: '🚫 Foods to Avoid', color: _currentPlan.primaryColor,
      child: Column(
        children: _currentPlan.foodsToAvoid.map((food) => Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(children: [
            Container(width: 8, height: 8, decoration: const BoxDecoration(color: Color(0xFFE53E3E), shape: BoxShape.circle)),
            const SizedBox(width: 12),
            Expanded(child: Text(food, style: const TextStyle(fontSize: 14, color: Color(0xFF444444)))),
          ]),
        )).toList(),
      ),
    );
  }

  Widget _buildWeeklyPlanSection() {
    return _buildSection(title: '📅 Weekly Meal Plan', color: _currentPlan.primaryColor,
      child: Column(children: [
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
        _buildMealCard('🌅 Breakfast', _currentDay.breakfast, const Color(0xFFFFF3CD), const Color(0xFFD97706)),
        const SizedBox(height: 12),
        _buildMealCard('☀️ Lunch', _currentDay.lunch, const Color(0xFFD1FAE5), const Color(0xFF059669)),
        const SizedBox(height: 12),
        _buildMealCard('🌙 Dinner', _currentDay.dinner, const Color(0xFFEDE9FE), const Color(0xFF7C3AED)),
        const SizedBox(height: 12),
        _buildMealCard('🍎 Snack', _currentDay.snack, const Color(0xFFFFE4E6), const Color(0xFFE11D48)),
      ]),
    );
  }

  Widget _buildMealCard(String label, Meal meal, Color bgColor, Color accentColor) {
    return Container(
      decoration: BoxDecoration(color: bgColor, borderRadius: BorderRadius.circular(16), border: Border.all(color: accentColor.withOpacity(0.3))),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
          leading: Container(padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(color: accentColor.withOpacity(0.15), borderRadius: BorderRadius.circular(10)),
              child: Text(label.split(' ')[0], style: const TextStyle(fontSize: 20))),
          title: Text(
            meal.name,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15, color: accentColor),
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  meal.description,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 12, color: accentColor.withOpacity(0.8)),
                ),
                const SizedBox(height: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(color: accentColor.withOpacity(0.15), borderRadius: BorderRadius.circular(10)),
                  child: Text(
                    meal.calories,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: accentColor),
                  ),
                ),
              ],
            ),
          ),
          children: [
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Ingredients & Portions:', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13, color: accentColor)),
              const SizedBox(height: 8),
              ...meal.items.map((item) => Padding(
                padding: const EdgeInsets.only(bottom: 5),
                child: Row(children: [
                  Icon(Icons.check_circle_outline, size: 16, color: accentColor),
                  const SizedBox(width: 8),
                  Expanded(child: Text(item, style: TextStyle(fontSize: 13, color: accentColor.withOpacity(0.9)))),
                ]),
              )),
            ]),
          ],
        ),
      ),
    );
  }

  Widget _buildNutrientsSection() {
    return _buildSection(title: '💊 Key Nutrients', color: _currentPlan.primaryColor,
      child: GridView.count(
        shrinkWrap: true, physics: const NeverScrollableScrollPhysics(),
        crossAxisCount: 2, mainAxisSpacing: 10, crossAxisSpacing: 10, childAspectRatio: 3,
        children: _currentPlan.keyNutrients.map((nutrient) => Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(color: _currentPlan.accentColor, borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _currentPlan.primaryColor.withOpacity(0.2))),
          child: Row(children: [
            Icon(Icons.local_pharmacy_outlined, size: 16, color: _currentPlan.primaryColor),
            const SizedBox(width: 6),
            Expanded(child: Text(nutrient,
                style: TextStyle(color: _currentPlan.primaryColor, fontWeight: FontWeight.w600, fontSize: 12),
                overflow: TextOverflow.ellipsis)),
          ]),
        )).toList(),
      ),
    );
  }

  Widget _buildExerciseSection() {
    final plan = _currentPlan;
    final workout = _currentWorkout;
    return _buildSection(title: '🏋️ Exercise & Workout Plan', color: plan.primaryColor,
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: plan.accentColor, borderRadius: BorderRadius.circular(14),
              border: Border.all(color: plan.primaryColor.withOpacity(0.25))),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Icon(Icons.info_outline_rounded, color: plan.primaryColor, size: 18),
            const SizedBox(width: 10),
            Expanded(child: Text(plan.exerciseOverview,
                style: TextStyle(fontSize: 13, color: plan.primaryColor.withOpacity(0.85), height: 1.5))),
          ]),
        ),
        const SizedBox(height: 16),
        Text('💡 Exercise Tips', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: plan.primaryColor)),
        const SizedBox(height: 8),
        ...plan.exerciseTips.map((tip) => Padding(
          padding: const EdgeInsets.only(bottom: 7),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(margin: const EdgeInsets.only(top: 5), width: 7, height: 7,
                decoration: BoxDecoration(color: plan.primaryColor, shape: BoxShape.circle)),
            const SizedBox(width: 10),
            Expanded(child: Text(tip, style: const TextStyle(fontSize: 13, color: Color(0xFF444444), height: 1.4))),
          ]),
        )),
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
                  child: Text(plan.workoutWeek[index].day,
                      style: TextStyle(color: isSelected ? Colors.white : plan.primaryColor, fontWeight: FontWeight.w700, fontSize: 13)),
                ),
              );
            },
          ),
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: [plan.primaryColor, plan.primaryColor.withOpacity(0.7)]),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.flash_on_rounded, color: Colors.white, size: 16),
            const SizedBox(width: 6),
            Text('Focus: ${workout.focus}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 13)),
          ]),
        ),
        const SizedBox(height: 14),
        ...workout.exercises.map((exercise) => _buildExerciseCard(exercise, plan.primaryColor, plan.accentColor)),
      ]),
    );
  }

  Widget _buildExerciseCard(Exercise exercise, Color primary, Color accent) {
    final intensityColor = _intensityColor(exercise.intensity);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: accent, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: primary.withOpacity(0.2), width: 1.5),
        boxShadow: [BoxShadow(color: primary.withOpacity(0.06), blurRadius: 8, offset: const Offset(0, 3))],
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.fromLTRB(14, 6, 14, 6),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
          leading: Container(padding: const EdgeInsets.all(9),
              decoration: BoxDecoration(color: primary.withOpacity(0.12), borderRadius: BorderRadius.circular(11)),
              child: Icon(exercise.icon, color: primary, size: 20)),
          title: Text(exercise.name, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: primary)),
          subtitle: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const SizedBox(height: 4),
            Row(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(color: primary.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.timer_outlined, size: 11, color: primary),
                  const SizedBox(width: 4),
                  Text(exercise.duration, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: primary)),
                ]),
              ),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(color: intensityColor.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.bolt_rounded, size: 11, color: intensityColor),
                  const SizedBox(width: 3),
                  Text(exercise.intensity, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: intensityColor)),
                ]),
              ),
            ]),
            const SizedBox(height: 5),
            Text(exercise.benefit, style: TextStyle(fontSize: 11.5, color: primary.withOpacity(0.75), height: 1.3)),
          ]),
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: primary.withOpacity(0.15))),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('How to do it:', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 12, color: primary)),
                const SizedBox(height: 8),
                ...List.generate(exercise.steps.length, (i) => Padding(
                  padding: const EdgeInsets.only(bottom: 7),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Container(width: 20, height: 20,
                        decoration: BoxDecoration(color: primary, shape: BoxShape.circle),
                        child: Center(child: Text('${i + 1}', style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)))),
                    const SizedBox(width: 10),
                    Expanded(child: Text(exercise.steps[i], style: const TextStyle(fontSize: 13, color: Color(0xFF333333), height: 1.4))),
                  ]),
                )),
              ]),
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
        color: Colors.white, borderRadius: BorderRadius.circular(18),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Container(width: 4, height: 20, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
          const SizedBox(width: 10),
          Text(title, style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: color)),
        ]),
        const SizedBox(height: 16),
        child,
      ]),
    );
  }
}