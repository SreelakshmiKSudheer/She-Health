import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/services.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz;
import 'package:shehealth/app_navigator.dart';

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

    await _notifications.initialize(
  settings,
  onDidReceiveNotificationResponse: (response) {
    final payload = response.payload;

    if (payload != null) {
      NotificationService.handleNavigation(payload);
    }
  },
);
  }

  static const AndroidNotificationDetails commonDetails =
    AndroidNotificationDetails(
  'health_channel',
  'Health Notifications',
  channelDescription: 'Daily women health notifications',
  importance: Importance.max,
  priority: Priority.high,
  styleInformation: BigTextStyleInformation(''),
);

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

  const details = NotificationDetails(android: commonDetails);

  await _notifications.show(
    0,
    title,
    body,
    details,
  );
}

static void handleNavigation(String payload) {
  switch (payload) {
    case "diet":
      navigatorKey.currentState?.pushNamed('/diet');
      break;

    case "workout":
      navigatorKey.currentState?.pushNamed('/workout');
      break;

    case "tip":
      navigatorKey.currentState?.pushNamed('/tips');
      break;

    case "reminder":
      navigatorKey.currentState?.pushNamed('/reminders');
      break;
  }
}

  static Future<void> _scheduleWithFallback({
    required int id,
    required String title,
    required String body,
    required tz.TZDateTime scheduledDate,
    required NotificationDetails notificationDetails,
    required String payload,
    DateTimeComponents? matchDateTimeComponents,
  }) async {
    Future<void> schedule(AndroidScheduleMode androidScheduleMode) async {
      await _notifications.zonedSchedule(
        id,
        title,
        body,
        scheduledDate,
        notificationDetails,
        payload: payload,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
        androidScheduleMode: androidScheduleMode,
        matchDateTimeComponents: matchDateTimeComponents,
      );
    }

    try {
      await schedule(AndroidScheduleMode.exactAllowWhileIdle);
    } on PlatformException catch (e) {
      final errorCode = e.code.toLowerCase();
      if (errorCode == 'exact_alarms_not_permitted' ||
          errorCode.contains('exact_alarm')) {
        await schedule(AndroidScheduleMode.inexactAllowWhileIdle);
        return;
      }
      rethrow;
    }
  }

  static Future<void> scheduleDietPlan(String dietSummary) async {
  await _scheduleWithFallback(
    id: 1, // ✅ UNIQUE ID
    title: "🥗 Today's Diet Plan",
    body: dietSummary,
    scheduledDate: _scheduleTime(7, 30),
    notificationDetails: const NotificationDetails(
      android: AndroidNotificationDetails(
        'diet_channel',
        'Diet Plans',
        importance: Importance.max,
        priority: Priority.high,
        styleInformation: BigTextStyleInformation('Follow your healthy diet plan today.'),
      ),
    ),
    payload: "diet",
    matchDateTimeComponents: DateTimeComponents.time,
  );
}

  // ✅ WORKOUT
  static Future<void> scheduleWorkout(String workout) async {
  await _scheduleWithFallback(
    id: 4,
    title: "💪 Workout Time",
    body: workout,
    scheduledDate: _scheduleTime(18, 0),
    notificationDetails: const NotificationDetails(
      android: AndroidNotificationDetails(
        'workout_channel',
        'Workout',
        importance: Importance.max,
        priority: Priority.high,
        styleInformation: BigTextStyleInformation('Stay active and strong.'),
      ),
    ),
    payload: "workout",
    matchDateTimeComponents: DateTimeComponents.time,
  );
}

  static Future<void> scheduleHealthTip(String tip) async {
  await _scheduleWithFallback(
    id: 3,
    title: "💡 Daily Health Tip",
    body: tip,
    scheduledDate: _scheduleTime(9, 0),
    notificationDetails: const NotificationDetails(
      android: AndroidNotificationDetails(
        'tip_channel',
        'Health Tips',
        importance: Importance.max,
        priority: Priority.high,
        styleInformation: BigTextStyleInformation('Small habits create big health improvements.'),
      ),
    ),
    payload: "tip",
    matchDateTimeComponents: DateTimeComponents.time,
  );
}

  static Future<void> scheduleReminder(String reminderText) async {
  await _scheduleWithFallback(
    id: 2, // ✅ UNIQUE
    title: "⏰ Today's Reminders",
    body: reminderText,
    scheduledDate: _scheduleTime(8, 0),
    notificationDetails: const NotificationDetails(
      android: AndroidNotificationDetails(
        'reminder_channel',
        'Daily Reminders',
        importance: Importance.max,
        priority: Priority.high,
        styleInformation: BigTextStyleInformation('Stay on track with your tasks today.'),
      ),
    ),
    payload: "reminder",
    matchDateTimeComponents: DateTimeComponents.time,
  );
}
}