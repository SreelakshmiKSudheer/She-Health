import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite/sqflite.dart';

import '../models/app_models.dart';

class LocalStorageService {
  LocalStorageService._();

  static final LocalStorageService instance = LocalStorageService._();

  Database? _db;
  static const String _usersPrefsKey = 'users_local_v1';

  static const Map<String, String> _usersTableColumns = {
    'user_id': 'TEXT PRIMARY KEY',
    'full_name': 'TEXT NOT NULL',
    'email': 'TEXT NOT NULL UNIQUE',
    'phone': 'TEXT NOT NULL',
    'password': 'TEXT NOT NULL',
    'dob': 'TEXT',
    'blood_group': 'TEXT',
    'marital_status': 'TEXT',
    'activity_level': 'TEXT',
    'emergency_contact': 'TEXT',
    'has_allergies': 'INTEGER NOT NULL DEFAULT 0',
    'has_chronic_conditions': 'INTEGER NOT NULL DEFAULT 0',
    'is_on_medication': 'INTEGER NOT NULL DEFAULT 0',
    'height_cm': 'REAL',
    'weight_kg': 'REAL',
  };

  Future<Database> get database async {
    if (_db != null) {
      return _db!;
    }

    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dbPath, 'shehealth_local.db'),
      version: 2,
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

        await _ensureUsersTableSchema(db);
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        await _ensureUsersTableSchema(db);
      },
      onOpen: (db) async {
        await _ensureUsersTableSchema(db);
      },
    );

    return _db!;
  }

  Future<void> _ensureUsersTableSchema(Database db) async {
    final tableInfo = await db.rawQuery('PRAGMA table_info(users_local)');
    final existingColumns = tableInfo
        .map((row) => row['name'] as String?)
        .whereType<String>()
        .toSet();

    for (final entry in _usersTableColumns.entries) {
      if (existingColumns.contains(entry.key)) {
        continue;
      }

      await db.execute(
        'ALTER TABLE users_local ADD COLUMN ${entry.key} ${entry.value}',
      );
    }
  }

  Future<void> upsertUser(LocalUserProfile user) async {
    if (kIsWeb) {
      await _upsertUserWeb(user);
      return;
    }

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
    if (kIsWeb) {
      return _findByEmailAndPasswordWeb(email: email, password: password);
    }

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
    if (kIsWeb) {
      return _findByUserIdWeb(userId);
    }

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

  Future<void> _upsertUserWeb(LocalUserProfile user) async {
    final users = await _readUsersFromPrefs();

    users.removeWhere((entry) {
      final entryUserId = entry['user_id']?.toString();
      final entryEmail = entry['email']?.toString();
      return entryUserId == user.userId || entryEmail == user.email;
    });

    users.add(Map<String, dynamic>.from(user.toMap()));
    await _writeUsersToPrefs(users);
  }

  Future<LocalUserProfile?> _findByEmailAndPasswordWeb({
    required String email,
    required String password,
  }) async {
    final users = await _readUsersFromPrefs();
    for (final entry in users) {
      if (entry['email']?.toString() == email &&
          entry['password']?.toString() == password) {
        return LocalUserProfile.fromMap(Map<String, Object?>.from(entry));
      }
    }
    return null;
  }

  Future<LocalUserProfile?> _findByUserIdWeb(String userId) async {
    final users = await _readUsersFromPrefs();
    for (final entry in users) {
      if (entry['user_id']?.toString() == userId) {
        return LocalUserProfile.fromMap(Map<String, Object?>.from(entry));
      }
    }
    return null;
  }

  Future<List<Map<String, dynamic>>> _readUsersFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_usersPrefsKey);
    if (raw == null || raw.isEmpty) {
      return <Map<String, dynamic>>[];
    }

    final decoded = jsonDecode(raw);
    if (decoded is! List) {
      return <Map<String, dynamic>>[];
    }

    return decoded
        .whereType<Map>()
        .map((entry) => Map<String, dynamic>.from(entry))
        .toList();
  }

  Future<void> _writeUsersToPrefs(List<Map<String, dynamic>> users) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_usersPrefsKey, jsonEncode(users));
  }
}
