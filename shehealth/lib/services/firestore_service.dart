import 'package:cloud_firestore/cloud_firestore.dart';

class FirestoreService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  Future<void> saveMessage({
    required String chatId,
    required String role,
    required String content,
  }) async {
    await _db
        .collection('chats')
        .doc(chatId)
        .collection('messages')
        .add({
      'role': role,
      'content': content,
      'timestamp': FieldValue.serverTimestamp(),
    });
  }

  Stream<QuerySnapshot> getMessages(String chatId) {
    return _db
        .collection('chats')
        .doc(chatId)
        .collection('messages')
        .orderBy('timestamp')
        .snapshots();
  }
}
