import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../models/app_models.dart';

class LocalStorageService {
  LocalStorageService._();

  static final LocalStorageService instance = LocalStorageService._();

  Database? _db;

  Future<Database> get database async {
    if (_db != null) {
      return _db!;
    }

    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dbPath, 'shehealth_local.db'),
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE users_local (
            user_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            dob TEXT,
            blood_group TEXT,
            marital_status TEXT,
            activity_level TEXT,
            emergency_contact TEXT,
            has_allergies INTEGER NOT NULL DEFAULT 0,
            has_chronic_conditions INTEGER NOT NULL DEFAULT 0,
            is_on_medication INTEGER NOT NULL DEFAULT 0,
            height_cm REAL,
            weight_kg REAL
          )
        ''');
      },
    );

    return _db!;
  }

  Future<void> upsertUser(LocalUserProfile user) async {
    final db = await database;
    await db.insert(
      'users_local',
      user.toMap(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<LocalUserProfile?> findByEmailAndPassword({
    required String email,
    required String password,
  }) async {
    final db = await database;
    final rows = await db.query(
      'users_local',
      where: 'email = ? AND password = ?',
      whereArgs: [email, password],
      limit: 1,
    );

    if (rows.isEmpty) {
      return null;
    }
    return LocalUserProfile.fromMap(rows.first);
  }

  Future<LocalUserProfile?> findByUserId(String userId) async {
    final db = await database;
    final rows = await db.query(
      'users_local',
      where: 'user_id = ?',
      whereArgs: [userId],
      limit: 1,
    );

    if (rows.isEmpty) {
      return null;
    }
    return LocalUserProfile.fromMap(rows.first);
  }
}
