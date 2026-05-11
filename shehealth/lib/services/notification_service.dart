import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz;

class NotificationService {
  static final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();

  // ✅ INIT
  static Future<void> init() async {
    const androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const settings = InitializationSettings(
      android: androidSettings,
    );

    tz.initializeTimeZones();
    tz.setLocalLocation(tz.getLocation('Asia/Kolkata'));

    await _notifications.initialize(settings);
  }

  // ✅ INSTANT NOTIFICATION
  static Future<void> showNotification({
    required String title,
    required String body,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'channel_id',
      'Health Notifications',
      importance: Importance.max,
      priority: Priority.high,
    );

    const details = NotificationDetails(android: androidDetails);

    await _notifications.show(
      0,
      title,
      body,
      details,
    );
  }

  // ✅ COMMON TIME SCHEDULER
  static tz.TZDateTime _scheduleTime(int hour, int minute) {
    final now = tz.TZDateTime.now(tz.local);

    var scheduled = tz.TZDateTime(
      tz.local,
      now.year,
      now.month,
      now.day,
      hour,
      minute,
    );

    if (scheduled.isBefore(now)) {
      scheduled = scheduled.add(const Duration(days: 1));
    }

    return scheduled;
  }

  static Future<void> showInstantNotification(
    String title, String body) async {
  await _notifications.show(
    999,
    title,
    body,
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'test_channel',
        'Test Notifications',
        importance: Importance.max,
        priority: Priority.high,
      ),
    ),
  );
}

  static Future<void> scheduleDietPlan(String dietSummary) async {
  await _notifications.zonedSchedule(
    3,
    "Today's Diet Plan 🥗",
    dietSummary,
    _scheduleTime(7, 30),
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'diet_channel',
        'Diet Plans',
      ),
    ),
    uiLocalNotificationDateInterpretation:
        UILocalNotificationDateInterpretation.absoluteTime,
    androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
    matchDateTimeComponents: DateTimeComponents.time,
  );
}

  // ✅ WORKOUT
  static Future<void> scheduleWorkout(String workout) async {
    await _notifications.zonedSchedule(
  2,
  "Workout Time 💪",
  workout,
  _scheduleTime(18, 0),
  const NotificationDetails(
    android: AndroidNotificationDetails(
      'workout_channel',
      'Workout',
    ),
  ),
  uiLocalNotificationDateInterpretation:
    UILocalNotificationDateInterpretation.absoluteTime,
  androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle, // ✅ FIX
  matchDateTimeComponents: DateTimeComponents.time,
);
  }

  static Future<void> scheduleHealthTip(String tip) async {
  await _notifications.zonedSchedule(
    3,
    "Health Tip 💡",
    tip, // ✅ dynamic from LLM
    _scheduleTime(9, 0),
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'tip_channel',
        'Health Tips',
      ),
    ),
    uiLocalNotificationDateInterpretation:
        UILocalNotificationDateInterpretation.absoluteTime,
    androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
    matchDateTimeComponents: DateTimeComponents.time,
  );
}

static Future<void> scheduleReminder(String reminderText) async {
  await _notifications.zonedSchedule(
    2,
    "Today's Reminders ⏰",
    reminderText,
    _scheduleTime(8, 0),
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'reminder_channel',
        'Daily Reminders',
      ),
    ),
    uiLocalNotificationDateInterpretation:
        UILocalNotificationDateInterpretation.absoluteTime,
    androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
    matchDateTimeComponents: DateTimeComponents.time,
  );
}
}