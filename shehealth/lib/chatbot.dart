import 'package:flutter/material.dart';
import 'services/groq_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class HealthChatbotPage extends StatefulWidget {
  const HealthChatbotPage({Key? key}) : super(key: key);

  @override
  State<HealthChatbotPage> createState() => _HealthChatbotPageState();
}

class _HealthChatbotPageState extends State<HealthChatbotPage>
    with SingleTickerProviderStateMixin {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  late AnimationController _pulseController;
  late GroqService _groqService;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  String? chatId = "default_chat";
  List<Map<String, dynamic>> chatMessages = [];
  List<Map<String, dynamic>> conversationHistory = [];
  bool _isTyping = false;

  final List<String> quickSuggestions = [
    'What is PCOS?',
    'Period irregularities',
    'Cervical cancer screening',
    'Endometriosis symptoms',
    'Fertility tips',
    'Healthy diet for women',
    'Exercise for hormonal balance',
    'Thyroid and women\'s health',
  ];

  final Map<String, String> healthResponses = {
    'period':
        'A normal menstrual cycle ranges from 21-35 days. If you experience irregularities, severe pain, or heavy bleeding, it\'s best to consult a gynecologist.',
    'pcos':
        'PCOS (Polycystic Ovary Syndrome) is a hormonal disorder. Common symptoms include irregular periods, excess hair growth, acne, and weight gain. Lifestyle changes and medication can help manage it.',
    'pregnancy':
        'If you suspect pregnancy, take a home pregnancy test and consult your doctor. Early prenatal care is important for a healthy pregnancy.',
  };

  @override
  void initState() {
    super.initState();
    _groqService = GroqService();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    // Load messages and show welcome
    _loadMessagesFromFirestore();
    Future.delayed(const Duration(milliseconds: 500), () {
      _addBotMessage(
        'Hello! 👋 I\'m your AI-powered women\'s health assistant. How can I help you today?',
        addToHistory: false,
      );
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _scrollController.dispose();
    _messageController.dispose();
    super.dispose();
  }

  Future<void> _loadMessagesFromFirestore() async {
    if (chatId == null) return;
    try {
      final snapshot = await _firestore
          .collection('chats')
          .doc(chatId)
          .collection('messages')
          .orderBy('timestamp')
          .get();

      setState(() {
        chatMessages = snapshot.docs.map((d) {
          final data = d.data() as Map<String, dynamic>;
          final ts = data['timestamp'];
          final DateTime timestamp =
              ts is Timestamp ? ts.toDate() : DateTime.now();
          return {
            'type': data['type'] as String? ?? 'bot',
            'message': data['message'] as String? ?? '',
            'timestamp': timestamp,
          };
        }).toList();

        conversationHistory = chatMessages.map((m) {
          return {
            'role': m['type'] == 'user' ? 'user' : 'assistant',
            'content': m['message'] ?? '',
          };
        }).toList();
      });
    } catch (_) {}
  }

  Future<void> _addBotMessage(String message,
      {bool addToHistory = true}) async {
    if (chatId == null) return;
    final entry = {
      'type': 'bot',
      'message': message,
      'timestamp': DateTime.now()
    };
    setState(() {
      chatMessages.add(entry);
      if (addToHistory) {
        conversationHistory.add({'role': 'assistant', 'content': message});
        if (conversationHistory.length > 20) {
          conversationHistory =
              conversationHistory.sublist(conversationHistory.length - 20);
        }
      }
    });

    try {
      await _firestore
          .collection('chats')
          .doc(chatId)
          .collection('messages')
          .add({
        'type': 'bot',
        'message': message,
        'timestamp': FieldValue.serverTimestamp(),
      });
    } catch (_) {}

    _scrollToBottom();
  }

  Future<void> _addUserMessage(String message) async {
    if (chatId == null) return;
    final entry = {
      'type': 'user',
      'message': message,
      'timestamp': DateTime.now()
    };
    setState(() {
      chatMessages.add(entry);
      conversationHistory.add({'role': 'user', 'content': message});
    });

    try {
      await _firestore
          .collection('chats')
          .doc(chatId)
          .collection('messages')
          .add({
        'type': 'user',
        'message': message,
        'timestamp': FieldValue.serverTimestamp(),
      });
    } catch (_) {}

    _scrollToBottom();
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 120), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  String _getBotResponse(String userMessage) {
    final msg = userMessage.toLowerCase();
    if (msg.contains('hello') || msg.contains('hi'))
      return 'Hello! How can I help?';
    for (final e in healthResponses.entries) {
      if (msg.contains(e.key)) return e.value;
    }
    return 'I\'m here to help with women\'s health topics. Try asking about PCOS, periods, pregnancy, or diet.';
  }

  Future<void> _handleSendMessage(String message) async {
    if (message.trim().isEmpty) return;

    if (chatMessages.isEmpty) {
      chatId = DateTime.now().millisecondsSinceEpoch.toString();
      try {
        await _firestore.collection('chats').doc(chatId).set({
          'title': message.length > 30 ? message.substring(0, 30) : message,
          'createdAt': FieldValue.serverTimestamp(),
        });
      } catch (_) {}
    }

    await _addUserMessage(message);
    _messageController.clear();

    setState(() => _isTyping = true);

    try {
      final historyForGroq = conversationHistory
          .map((m) => {
                'role': m['role'].toString(),
                'content': m['content'].toString()
              })
          .toList();

      final response = await _groqService.sendMessage(message, historyForGroq);
      setState(() => _isTyping = false);
      await _addBotMessage(response);
    } catch (_) {
      setState(() => _isTyping = false);
      await _addBotMessage('Sorry, I\'m having trouble connecting right now.',
          addToHistory: false);
    }
  }

  void _handleQuickSuggestion(String suggestion) =>
      _handleSendMessage(suggestion);

  Future<void> _clearConversation() async {
    if (chatId != null) {
      final snapshot = await _firestore
          .collection('chats')
          .doc(chatId)
          .collection('messages')
          .get();
      for (var doc in snapshot.docs) {
        await doc.reference.delete();
      }
    }
    setState(() {
      chatMessages.clear();
      conversationHistory.clear();
      chatId = null;
    });
    Future.delayed(const Duration(milliseconds: 300), () {
      _addBotMessage('Chat cleared! How can I help you today?',
          addToHistory: false);
    });
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: const BoxDecoration(
        gradient:
            LinearGradient(colors: [Color(0xFFC85A7A), Color(0xFFE59393)]),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Scaffold.of(context).openDrawer(),
            icon: const Icon(Icons.menu, color: Colors.white),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text('Health Assistant',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold)),
                SizedBox(height: 4),
                Text('AI • Women\'s health',
                    style: TextStyle(color: Colors.white70, fontSize: 12)),
              ],
            ),
          ),
          IconButton(
              onPressed: _clearConversation,
              icon: const Icon(Icons.delete_outline, color: Colors.white)),
        ],
      ),
    );
  }

  Widget _buildChatBubble(Map<String, dynamic> message) {
    final bool isBot = message['type'] == 'bot';
    final text = message['message']?.toString() ?? '';
    return Align(
      alignment: isBot ? Alignment.centerLeft : Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(12),
        constraints:
            BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
        decoration: BoxDecoration(
          color: isBot ? const Color(0xFFF2F2F2) : const Color(0xFFC85A7A),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          text,
          style:
              TextStyle(color: isBot ? const Color(0xFF333333) : Colors.white),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
                color: const Color(0xFFFFF5F8),
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0xFFE5C4C4), width: 2)),
            child: const Icon(Icons.health_and_safety,
                size: 48, color: Color(0xFFC85A7A)),
          ),
          const SizedBox(height: 20),
          const Text('Welcome to Health Assistant',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 32),
            child: Text(
                'Ask me anything about women\'s health — try a quick suggestion below.',
                textAlign: TextAlign.center),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickSuggestions() {
    return SizedBox(
      height: 48,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        scrollDirection: Axis.horizontal,
        itemBuilder: (context, index) {
          final s = quickSuggestions[index];
          return ActionChip(
              label: Text(s), onPressed: () => _handleQuickSuggestion(s));
        },
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemCount: quickSuggestions.length,
      ),
    );
  }

  Widget _buildMessageInput() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.white,
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _messageController,
              textInputAction: TextInputAction.send,
              onSubmitted: (v) => _handleSendMessage(v),
              decoration: const InputDecoration(
                  hintText: 'Ask a question...',
                  border: OutlineInputBorder(),
                  contentPadding:
                      EdgeInsets.symmetric(horizontal: 12, vertical: 8)),
            ),
          ),
          const SizedBox(width: 8),
          ElevatedButton(
            onPressed: () => _handleSendMessage(_messageController.text),
            child: const Icon(Icons.send),
            style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC85A7A)),
          ),
        ],
      ),
    );
  }

  Widget _buildTypingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
            color: const Color(0xFFF2F2F2),
            borderRadius: BorderRadius.circular(12)),
        child: const Text('Typing...'),
      ),
    );
  }

  Widget _buildChatDrawer() {
    return Drawer(
      child: SafeArea(
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: const BoxDecoration(
                  gradient: LinearGradient(
                      colors: [Color(0xFFC85A7A), Color(0xFFE59393)])),
              child: const Text('Your Chats',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold)),
            ),
            Expanded(
              child: StreamBuilder<QuerySnapshot>(
                stream: _firestore
                    .collection('chats')
                    .orderBy('createdAt', descending: true)
                    .snapshots(),
                builder: (context, snapshot) {
                  if (!snapshot.hasData)
                    return const Center(child: CircularProgressIndicator());
                  final docs = snapshot.data!.docs;
                  if (docs.isEmpty)
                    return const Center(child: Text('No chats yet'));
                  return ListView.builder(
                    itemCount: docs.length,
                    itemBuilder: (context, index) {
                      final d = docs[index];
                      final data = d.data() as Map<String, dynamic>;
                      final title = data['title'] as String? ?? 'Chat';
                      return ListTile(
                        title: Text(title),
                        subtitle: data['createdAt'] != null
                            ? Text((data['createdAt'] as Timestamp)
                                .toDate()
                                .toString())
                            : null,
                        onTap: () {
                          Navigator.pop(context);
                          setState(() {
                            chatId = d.id;
                            chatMessages.clear();
                            conversationHistory.clear();
                          });
                          _loadMessagesFromFirestore();
                        },
                        trailing: IconButton(
                            icon: const Icon(Icons.delete_outline),
                            onPressed: () async {
                              await _deleteChat(d.id);
                            }),
                      );
                    },
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(12),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () async {
                    Navigator.pop(context);
                    setState(() {
                      chatId = null;
                      chatMessages.clear();
                      conversationHistory.clear();
                    });
                    Future.delayed(const Duration(milliseconds: 300), () {
                      _addBotMessage('Hello! 👋 How can I help you today?',
                          addToHistory: false);
                    });
                  },
                  icon: const Icon(Icons.add),
                  label: const Text('New Chat'),
                  style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFC85A7A)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _deleteChat(String id) async {
    final snapshot = await _firestore
        .collection('chats')
        .doc(id)
        .collection('messages')
        .get();
    for (var doc in snapshot.docs) await doc.reference.delete();
    await _firestore.collection('chats').doc(id).delete();
    if (id == chatId) {
      setState(() {
        chatId = null;
        chatMessages.clear();
        conversationHistory.clear();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: _buildChatDrawer(),
      appBar: AppBar(
          title: const Text('Health Assistant'),
          backgroundColor: const Color(0xFFC85A7A)),
      body: Column(
        children: [
          _buildHeader(),
          Expanded(
            child: chatMessages.isEmpty
                ? _buildEmptyState()
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: chatMessages.length + (_isTyping ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (_isTyping && index == chatMessages.length)
                        return _buildTypingIndicator();
                      return _buildChatBubble(chatMessages[index]);
                    },
                  ),
          ),
          _buildQuickSuggestions(),
          _buildMessageInput(),
        ],
      ),
    );
  }
}
