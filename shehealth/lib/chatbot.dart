import 'package:flutter/material.dart';
<<<<<<< Updated upstream
import 'services/groq_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
=======
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
  late GroqService _groqService;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // FIX 1: Made non-final and nullable so it can be reassigned
  String? chatId = "default_chat";

  // FIX 2: Explicit Map<String, dynamic> types to avoid inference errors
  List<Map<String, dynamic>> chatMessages = [];
  List<Map<String, dynamic>> conversationHistory = [];

  bool _isTyping = false;

  Future<void> _loadMessagesFromFirestore() async {
    if (chatId == null) return;

    final snapshot = await _firestore
        .collection('chats')
        .doc(chatId)
        .collection('messages')
        .orderBy('timestamp')
        .get();

    setState(() {
      chatMessages = snapshot.docs.map((doc) {
        final data = doc.data();
        // FIX 3: Convert Firestore Timestamp to DateTime safely
        final ts = data['timestamp'];
        final DateTime dateTime =
            ts is Timestamp ? ts.toDate() : DateTime.now();
        return <String, dynamic>{
          'type': data['type'] as String? ?? 'bot',
          'message': data['message'] as String? ?? '',
          'timestamp': dateTime,
        };
      }).toList();

      // FIX 4: Explicit cast avoids Map<String,String> inference mismatch
      conversationHistory = chatMessages.map((msg) {
        return <String, dynamic>{
          "role": msg['type'] == 'user' ? "user" : "assistant",
          "content": msg['message'],
        };
      }).toList();
    });
  }

  // Quick suggestion chips
  final List<String> quickSuggestions = [
    'What is PCOS?',
    'Period irregularities',
    'Cervical cancer screening',
    'Endometriosis symptoms',
    'Fertility tips',
    'Healthy diet for women',
    'Exercise for hormonal balance',
    'Thyroid and women\'s health',
=======
  List<Map<String, dynamic>> chatMessages = [];
  bool _isTyping = false;

  // Predefined responses for common women's health queries
  final Map<String, String> healthResponses = {
    // Menstrual Health
    'period':
        'A normal menstrual cycle ranges from 21-35 days. If you experience irregularities, severe pain, or heavy bleeding, it\'s best to consult a gynecologist.',
    'cramps':
        'Menstrual cramps are common. Try heat therapy, light exercise, staying hydrated, and over-the-counter pain relievers. If pain is severe, consult your doctor.',
    'irregular cycle':
        'Irregular periods can be caused by stress, weight changes, PCOS, thyroid issues, or other factors. Track your cycle and consult a healthcare provider if it persists.',
    'heavy bleeding':
        'Heavy menstrual bleeding (menorrhagia) can indicate hormonal imbalances, fibroids, or other conditions. Please consult a gynecologist for proper evaluation.',

    // PCOS/PCOD
    'pcos':
        'PCOS (Polycystic Ovary Syndrome) is a hormonal disorder. Common symptoms include irregular periods, excess hair growth, acne, and weight gain. Lifestyle changes and medication can help manage it.',
    'pcod':
        'PCOD (Polycystic Ovarian Disease) affects hormone levels. Symptoms include irregular periods, weight gain, and acne. A balanced diet, exercise, and medical consultation can help.',

    // Pregnancy & Fertility
    'pregnancy':
        'If you suspect pregnancy, take a home pregnancy test and consult your doctor. Early prenatal care is important for a healthy pregnancy.',
    'fertility':
        'Fertility can be affected by age, lifestyle, health conditions, and other factors. Maintain a healthy weight, avoid smoking/alcohol, and consult a fertility specialist if trying to conceive.',
    'ovulation':
        'Ovulation typically occurs around day 14 of a 28-day cycle. You can track ovulation using apps, temperature monitoring, or ovulation prediction kits.',

    // General Health
    'diet':
        'A balanced diet rich in iron, calcium, folate, and omega-3 is important for women\'s health. Include fruits, vegetables, whole grains, lean proteins, and healthy fats.',
    'exercise':
        'Regular exercise (150 minutes/week) helps maintain hormonal balance, reduces stress, and improves overall health. Try yoga, walking, swimming, or strength training.',
    'stress':
        'Chronic stress can affect menstrual cycles and overall health. Practice stress management through meditation, yoga, deep breathing, adequate sleep, and seeking support.',

    // Thyroid
    'thyroid':
        'Thyroid disorders can affect menstruation, weight, energy, and mood. Symptoms include fatigue, weight changes, and hair loss. Get thyroid function tests done regularly.',

    // Screening & Prevention
    'pap smear':
        'Pap smears screen for cervical cancer. Women should start at age 21 and continue every 3 years (or as advised by doctor). It\'s a quick, important preventive test.',
    'breast exam':
        'Perform monthly breast self-exams and get clinical exams annually. Mammograms are recommended starting at age 40-50 (or earlier if at high risk).',
    'checkup':
        'Annual gynecological checkups are important for preventive care. They include pelvic exams, breast exams, and discussions about reproductive health.',

    // Lifestyle
    'sleep':
        'Aim for 7-9 hours of quality sleep. Good sleep regulates hormones, reduces stress, and supports overall health. Maintain a consistent sleep schedule.',
    'water':
        'Drink 8-10 glasses of water daily. Proper hydration helps reduce bloating, supports metabolism, and maintains healthy skin.',
    'vitamins':
        'Important vitamins for women include Vitamin D, B12, Iron, Calcium, and Folic Acid. Consult your doctor before starting any supplements.',

    // Pain Management
    'pain':
        'For menstrual pain, try heat therapy, gentle exercise, and OTC pain relievers. If pain is severe or interferes with daily activities, consult a doctor.',
    'endometriosis':
        'Endometriosis causes severe pelvic pain and heavy periods. It requires proper diagnosis and treatment. Consult a gynecologist if you suspect this condition.',

    // Mental Health
    'mood':
        'Hormonal fluctuations can affect mood. PMS, PMDD, or hormonal imbalances may cause mood swings. Track your symptoms and discuss with your doctor if severe.',
    'depression':
        'If you\'re experiencing persistent sadness, anxiety, or depression, please seek help from a mental health professional. Your mental health is just as important as physical health.',
  };

  // Quick suggestion chips
  final List<String> quickSuggestions = [
    'Period irregularities',
    'PCOS symptoms',
    'Fertility tips',
    'Healthy diet',
    'Exercise routine',
    'Stress management',
    'Thyroid issues',
    'Screening tests',
>>>>>>> Stashed changes
  ];

  @override
  void initState() {
    super.initState();
<<<<<<< Updated upstream
    _groqService = GroqService();
    _loadMessagesFromFirestore();
=======
>>>>>>> Stashed changes
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    // Add welcome message
    Future.delayed(const Duration(milliseconds: 500), () {
      _addBotMessage(
<<<<<<< Updated upstream
        'Hello! 👋 I\'m your AI-powered women\'s health assistant. I can help answer questions about:\n\n'
        '• PCOS/PCOD and hormonal health\n'
        '• Endometriosis\n'
        '• Cervical cancer prevention\n'
        '• Menstrual health\n'
        '• Pregnancy & fertility\n'
        '• General wellness\n\n'
        'How can I help you today?',
        addToHistory: false,
      );
=======
          'Hello! 👋 I\'m your women\'s health assistant. I\'m here to answer your questions about:\n\n'
          '• Menstrual health & cycles\n'
          '• PCOS/PCOD\n'
          '• Pregnancy & fertility\n'
          '• Diet & lifestyle\n'
          '• Thyroid health\n'
          '• Preventive screenings\n\n'
          'How can I help you today?');
>>>>>>> Stashed changes
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _scrollController.dispose();
    _messageController.dispose();
    super.dispose();
  }

<<<<<<< Updated upstream
  Future<void> _addBotMessage(String message,
      {bool addToHistory = true}) async {
    if (chatId == null) return;

    final Map<String, dynamic> messageData = {
      'type': 'bot',
      'message': message,
      'timestamp': DateTime.now(),
    };

    setState(() {
      chatMessages.add(messageData);
      if (addToHistory) {
        conversationHistory.add(<String, dynamic>{
          'role': 'assistant',
          'content': message,
        });
        if (conversationHistory.length > 20) {
          conversationHistory =
              conversationHistory.sublist(conversationHistory.length - 20);
        }
      }
    });

    await _firestore
        .collection('chats')
        .doc(chatId!)
        .collection('messages')
        .add({
      'type': 'bot',
      'message': message,
      'timestamp': FieldValue.serverTimestamp(),
    });

    _scrollToBottom();
  }

  Future<void> _addUserMessage(String message) async {
    if (chatId == null) return;

    final Map<String, dynamic> messageData = {
      'type': 'user',
      'message': message,
      'timestamp': DateTime.now(),
    };

    setState(() {
      chatMessages.add(messageData);
      conversationHistory.add(<String, dynamic>{
        'role': 'user',
        'content': message,
      });
    });

    await _firestore
        .collection('chats')
        .doc(chatId!)
        .collection('messages')
        .add({
      'type': 'user',
      'message': message,
      'timestamp': FieldValue.serverTimestamp(),
    });

=======
  void _addBotMessage(String message) {
    setState(() {
      chatMessages.add({
        'type': 'bot',
        'message': message,
        'timestamp': DateTime.now(),
      });
    });
    _scrollToBottom();
  }

  void _addUserMessage(String message) {
    setState(() {
      chatMessages.add({
        'type': 'user',
        'message': message,
        'timestamp': DateTime.now(),
      });
    });
>>>>>>> Stashed changes
    _scrollToBottom();
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

<<<<<<< Updated upstream
  Future<void> _handleSendMessage(String message) async {
    if (message.trim().isEmpty) return;

    // If this is a NEW chat (no messages yet)
    if (chatMessages.isEmpty) {
      chatId = DateTime.now().millisecondsSinceEpoch.toString();
      await _firestore.collection('chats').doc(chatId).set({
        'title': message.length > 30 ? message.substring(0, 30) : message,
        'createdAt': FieldValue.serverTimestamp(),
      });
    }

    await _addUserMessage(message);
    _messageController.clear();

=======
  String _getBotResponse(String userMessage) {
    String message = userMessage.toLowerCase().trim();

    // Check for greetings
    if (message.contains('hello') ||
        message.contains('hi') ||
        message.contains('hey')) {
      return 'Hello! How can I assist you with your women\'s health questions today? 😊';
    }

    if (message.contains('thank') || message.contains('thanks')) {
      return 'You\'re welcome! Feel free to ask if you have any other questions about your health. Take care! 💗';
    }

    if (message.contains('bye') || message.contains('goodbye')) {
      return 'Take care! Remember to prioritize your health and consult healthcare professionals for personalized advice. Stay healthy! 💖';
    }

    // Search for matching health topics
    for (var entry in healthResponses.entries) {
      if (message.contains(entry.key)) {
        return entry.value;
      }
    }

    // Default response if no match found
    return 'I understand you have a question about women\'s health. While I can provide general information, I recommend consulting with a healthcare professional for personalized advice.\n\n'
        'You can ask me about:\n'
        '• Menstrual health\n'
        '• PCOS/PCOD\n'
        '• Pregnancy & fertility\n'
        '• Diet & exercise\n'
        '• Thyroid health\n'
        '• Preventive care\n\n'
        'Please rephrase your question or choose from the suggestions below.';
  }

  void _handleSendMessage(String message) {
    if (message.trim().isEmpty) return;

    // Add user message
    _addUserMessage(message);
    _messageController.clear();

    // Show typing indicator
>>>>>>> Stashed changes
    setState(() {
      _isTyping = true;
    });

<<<<<<< Updated upstream
    try {
      // FIX: Convert List<Map<String, dynamic>> → List<Map<String, String>>
      final List<Map<String, String>> historyForGroq = conversationHistory
          .map((msg) => {
                'role': msg['role'].toString(),
                'content': msg['content'].toString(),
              })
          .toList();

      String response = await _groqService.sendMessage(
        message,
        historyForGroq,   // ✅ pass the converted list
      );
      setState(() {
        _isTyping = false;
      });
      await _addBotMessage(response);
    } catch (e) {
      setState(() {
        _isTyping = false;
      });
      await _addBotMessage(
        'I apologize for the inconvenience. I\'m having trouble processing your request. Please try again.',
        addToHistory: false,
      );
    }
=======
    // Simulate bot thinking and response
    Future.delayed(const Duration(milliseconds: 1500), () {
      setState(() {
        _isTyping = false;
      });
      String response = _getBotResponse(message);
      _addBotMessage(response);
    });
>>>>>>> Stashed changes
  }

  void _handleQuickSuggestion(String suggestion) {
    _handleSendMessage(suggestion);
  }

<<<<<<< Updated upstream
  void _clearConversation() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Conversation'),
        content:
            const Text('Are you sure you want to clear the chat history?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
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
              });
              Navigator.pop(context);
              Future.delayed(const Duration(milliseconds: 300), () {
                _addBotMessage(
                  'Chat cleared! How can I help you today?',
                  addToHistory: false,
                );
              });
            },
            child: const Text('Clear', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: _buildChatDrawer(),
=======
  @override
  Widget build(BuildContext context) {
    return Scaffold(
>>>>>>> Stashed changes
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
<<<<<<< Updated upstream
            _buildHeader(),
            Expanded(
              child: chatMessages.isEmpty
                  ? _buildEmptyState()
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.all(20),
                      itemCount: chatMessages.length + (_isTyping ? 1 : 0),
                      itemBuilder: (context, index) {
                        if (_isTyping && index == chatMessages.length) {
                          return _buildTypingIndicator();
                        }
                        return _buildChatBubble(chatMessages[index]);
                      },
                    ),
            ),
            if (chatMessages.length <= 1) _buildQuickSuggestions(),
=======
            // Header
            _buildHeader(),

            // Chat Messages
            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.all(20),
                itemCount: chatMessages.length + (_isTyping ? 1 : 0),
                itemBuilder: (context, index) {
                  if (_isTyping && index == chatMessages.length) {
                    return _buildTypingIndicator();
                  }
                  return _buildChatBubble(chatMessages[index]);
                },
              ),
            ),

            // Quick Suggestions
            if (chatMessages.length <= 1) _buildQuickSuggestions(),

            // Message Input
>>>>>>> Stashed changes
            _buildMessageInput(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
<<<<<<< Updated upstream
  return Container(
    decoration: const BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
      ),
    ),
    child: Stack(
      children: [
        Positioned(
          top: -30,
          right: -30,
          child: Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
        Positioned(
          bottom: -20,
          left: -20,
          child: Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              // ✅ Menu button to open drawer
              Builder(
                builder: (context) => IconButton(
                  onPressed: () => Scaffold.of(context).openDrawer(),
                  icon: const Icon(Icons.menu, color: Colors.white, size: 28),
                ),
              ),
              // ✅ Back button to go to previous screen
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.arrow_back,
                    color: Colors.white, size: 28),
              ),
              const SizedBox(width: 8),
              ScaleTransition(
                scale: Tween<double>(begin: 1.0, end: 1.05).animate(
                  CurvedAnimation(
                    parent: _pulseController,
                    curve: Curves.easeInOut,
                  ),
                ),
                child: Container(
                  width: 50,
                  height: 50,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.chat_bubble,
                    color: Color(0xFFC85A7A),
                    size: 24,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Health Assistant',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Color(0xFF4CAF50),
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          'AI Powered • Always here',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.9),
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              IconButton(
                onPressed: _clearConversation,
                icon: const Icon(Icons.delete_outline,
                    color: Colors.white, size: 24),
                tooltip: 'Clear chat',
              ),
            ],
          ),
        ),
      ],
    ),
  );
}
  Widget _buildChatDrawer() {
  return Drawer(
    child: SafeArea(
      child: Column(
        children: [
          // Header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
              ),
            ),
            child: const Text(
              "💬 Your Chats",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),

          // Chat list
          Expanded(
            child: StreamBuilder<QuerySnapshot>(
              stream: _firestore
                  .collection('chats')
                  .orderBy('createdAt', descending: true)
                  .snapshots(),
              builder: (context, snapshot) {
                if (!snapshot.hasData) {
                  return const Center(child: CircularProgressIndicator());
                }

                final chats = snapshot.data!.docs;

                if (chats.isEmpty) {
                  return const Center(
                    child: Text(
                      'No chats yet.\nStart a new conversation!',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.grey),
                    ),
                  );
                }

                return ListView.builder(
                  itemCount: chats.length,
                  itemBuilder: (context, index) {
                    final chat = chats[index];
                    final data = chat.data() as Map<String, dynamic>;
                    final title = data['title'] as String? ?? 'Untitled Chat';
                    final isActive = chat.id == chatId; // highlight active chat

                    return Container(
                      margin: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: isActive
                            ? const Color(0xFFC85A7A).withOpacity(0.15)
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: ListTile(
                        leading: Icon(
                          Icons.chat_bubble_outline,
                          color: isActive
                              ? const Color(0xFFC85A7A)
                              : Colors.grey,
                          size: 20,
                        ),
                        title: Text(
                          title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: isActive
                                ? FontWeight.bold
                                : FontWeight.normal,
                            color: isActive
                                ? const Color(0xFFC85A7A)
                                : const Color(0xFF333333),
                          ),
                        ),
                        // Show timestamp
                        subtitle: data['createdAt'] != null
                            ? Text(
                                _formatDrawerTime(
                                    (data['createdAt'] as Timestamp).toDate()),
                                style: const TextStyle(
                                    fontSize: 11, color: Colors.grey),
                              )
                            : null,
                        // Delete chat button
                        trailing: IconButton(
                          icon: const Icon(Icons.delete_outline,
                              size: 18, color: Colors.grey),
                          onPressed: () => _deleteChat(chat.id),
                        ),
                        onTap: () {
                          Navigator.pop(context);
                          setState(() {
                            chatId = chat.id;
                            chatMessages.clear();
                            conversationHistory.clear();
                          });
                          _loadMessagesFromFirestore();
                        },
                      ),
                    );
                  },
                );
              },
            ),
          ),

          // New Chat button
          Padding(
            padding: const EdgeInsets.all(12),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: () {
                  Navigator.pop(context);
                  setState(() {
                    chatId = null;
                    chatMessages.clear();
                    conversationHistory.clear();
                  });
                  // Show welcome message again
                  Future.delayed(const Duration(milliseconds: 300), () {
                    _addBotMessage(
                      'Hello! 👋 How can I help you today?',
                      addToHistory: false,
                    );
                  });
                },
                icon: const Icon(Icons.add),
                label: const Text('New Chat'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFC85A7A),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    ),
  );
}

Future<void> _deleteChat(String deleteChatId) async {
  // Delete all messages in the chat
  final messages = await _firestore
      .collection('chats')
      .doc(deleteChatId)
      .collection('messages')
      .get();

  for (var doc in messages.docs) {
    await doc.reference.delete();
  }

  // Delete the chat document itself
  await _firestore.collection('chats').doc(deleteChatId).delete();

  // If user deleted the active chat, start a new one
  if (deleteChatId == chatId) {
    setState(() {
      chatId = null;
      chatMessages.clear();
      conversationHistory.clear();
    });
  }
}

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(30),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF5F8),
              shape: BoxShape.circle,
              border: Border.all(
                color: const Color(0xFFE5C4C4),
                width: 2,
              ),
            ),
            child: const Icon(
              Icons.health_and_safety,
              size: 60,
              color: Color(0xFFC85A7A),
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'Welcome to Health Assistant',
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: Color(0xFF333333),
            ),
          ),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Text(
              'Ask me anything about women\'s health, and I\'ll provide helpful information.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
                height: 1.5,
              ),
=======
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
        ),
      ),
      child: Stack(
        children: [
          // Decorative Circles
          Positioned(
            top: -30,
            right: -30,
            child: Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
            ),
          ),
          Positioned(
            bottom: -20,
            left: -20,
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.arrow_back,
                      color: Colors.white, size: 28),
                ),
                const SizedBox(width: 8),
                ScaleTransition(
                  scale: Tween<double>(begin: 1.0, end: 1.05).animate(
                    CurvedAnimation(
                      parent: _pulseController,
                      curve: Curves.easeInOut,
                    ),
                  ),
                  child: Container(
                    width: 50,
                    height: 50,
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.chat_bubble,
                      color: Color(0xFFC85A7A),
                      size: 24,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Health Assistant',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: const BoxDecoration(
                              color: Color(0xFF4CAF50),
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            'Online • Always here to help',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.9),
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
>>>>>>> Stashed changes
            ),
          ),
        ],
      ),
    );
  }

<<<<<<< Updated upstream
  Widget _buildFormattedMessage(String text, bool isBot) {
    if (!isBot) {
      return Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 15,
          fontWeight: FontWeight.w600,
          height: 1.4,
        ),
      );
    }

    List<TextSpan> spans = [];
    final RegExp boldExp = RegExp(r'\*(.*?)\*');
    int currentIndex = 0;

    for (final match in boldExp.allMatches(text)) {
      if (match.start > currentIndex) {
        spans.add(TextSpan(
          text: text.substring(currentIndex, match.start),
          style: const TextStyle(
            fontSize: 15,
            color: Color(0xFF444444),
            height: 1.6,
          ),
        ));
      }
      spans.add(TextSpan(
        text: match.group(1),
        style: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.bold,
          color: Color(0xFF333333),
          height: 1.6,
        ),
      ));
      currentIndex = match.end;
    }

    if (currentIndex < text.length) {
      spans.add(TextSpan(
        text: text.substring(currentIndex),
        style: const TextStyle(
          fontSize: 15,
          color: Color(0xFF444444),
          height: 1.6,
        ),
      ));
    }

    return RichText(text: TextSpan(children: spans));
  }

  Widget _buildChatBubble(Map<String, dynamic> message) {
    bool isBot = message['type'] == 'bot';

    // FIX 5: Safely parse timestamp whether it's DateTime or Firestore Timestamp
    DateTime timestamp;
    final ts = message['timestamp'];
    if (ts is DateTime) {
      timestamp = ts;
    } else if (ts is Timestamp) {
      timestamp = ts.toDate();
    } else {
      timestamp = DateTime.now();
    }

=======
  Widget _buildChatBubble(Map<String, dynamic> message) {
    bool isBot = message['type'] == 'bot';

>>>>>>> Stashed changes
    return Align(
      alignment: isBot ? Alignment.centerLeft : Alignment.centerRight,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
        decoration: BoxDecoration(
          gradient: isBot
              ? LinearGradient(
                  colors: [
                    const Color(0xFFE5C4C4).withOpacity(0.4),
                    const Color(0xFFE59393).withOpacity(0.2),
                  ],
                )
              : const LinearGradient(
                  colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
                ),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(20),
            topRight: const Radius.circular(20),
            bottomLeft: Radius.circular(isBot ? 4 : 20),
            bottomRight: Radius.circular(isBot ? 20 : 4),
          ),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFFC85A7A).withOpacity(0.1),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
<<<<<<< Updated upstream
            _buildFormattedMessage(message['message'] as String, isBot),
            const SizedBox(height: 4),
            Text(
              _formatTime(timestamp),
=======
            Text(
              message['message'],
              style: TextStyle(
                color: isBot ? const Color(0xFF333333) : Colors.white,
                fontSize: 15,
                fontWeight: isBot ? FontWeight.w500 : FontWeight.w600,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              _formatTime(message['timestamp']),
>>>>>>> Stashed changes
              style: TextStyle(
                color: isBot
                    ? const Color(0xFF999999)
                    : Colors.white.withOpacity(0.7),
                fontSize: 11,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTypingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color(0xFFE5C4C4).withOpacity(0.4),
              const Color(0xFFE59393).withOpacity(0.2),
            ],
          ),
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
            bottomRight: Radius.circular(20),
            bottomLeft: Radius.circular(4),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildTypingDot(0),
            const SizedBox(width: 4),
            _buildTypingDot(200),
            const SizedBox(width: 4),
            _buildTypingDot(400),
          ],
        ),
      ),
    );
  }

  Widget _buildTypingDot(int delay) {
    return TweenAnimationBuilder<double>(
<<<<<<< Updated upstream
      tween: Tween<double>(begin: 0.0, end: 1.0),
=======
      tween: Tween(begin: 0.0, end: 1.0),
>>>>>>> Stashed changes
      duration: const Duration(milliseconds: 600),
      builder: (context, value, child) {
        return Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: Color.lerp(
              const Color(0xFFC85A7A).withOpacity(0.3),
              const Color(0xFFC85A7A),
              value,
            ),
            shape: BoxShape.circle,
          ),
        );
      },
      onEnd: () {
        setState(() {});
      },
    );
  }

  Widget _buildQuickSuggestions() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF5F8),
        border: Border(
          top: BorderSide(
            color: const Color(0xFFE5C4C4).withOpacity(0.5),
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Quick Suggestions:',
            style: TextStyle(
              color: Color(0xFF666666),
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: quickSuggestions.map((suggestion) {
              return InkWell(
                onTap: () => _handleQuickSuggestion(suggestion),
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    border: Border.all(
                      color: const Color(0xFFE5C4C4),
                      width: 1.5,
                    ),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    suggestion,
                    style: const TextStyle(
                      color: Color(0xFFC85A7A),
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageInput() {
    return Container(
<<<<<<< Updated upstream
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
=======
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          top: BorderSide(
            color: const Color(0xFFE5C4C4).withOpacity(0.5),
          ),
        ),
>>>>>>> Stashed changes
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
<<<<<<< Updated upstream
            child: TextField(
              controller: _messageController,
              maxLines: 4,
              minLines: 1,
              decoration: InputDecoration(
                hintText: "Ask me anything about women's health...",
                filled: true,
                fillColor: const Color(0xFFFFF5F8),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 18,
                  vertical: 14,
                ),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: const BorderSide(
                    color: Color(0xFFE5C4C4),
                    width: 1.5,
                  ),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: const BorderSide(
                    color: Color(0xFFE5C4C4),
                    width: 1.5,
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: const BorderSide(
                    color: Color(0xFFC85A7A),
                    width: 2,
                  ),
                ),
              ),
              textInputAction: TextInputAction.send,
              onSubmitted: _handleSendMessage,
            ),
          ),
          const SizedBox(width: 10),
          Container(
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
              ),
            ),
            child: IconButton(
              icon: const Icon(Icons.send, color: Colors.white),
              onPressed: () => _handleSendMessage(_messageController.text),
=======
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF5F8),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                  color: const Color(0xFFE5C4C4),
                  width: 1.5,
                ),
              ),
              child: TextField(
                controller: _messageController,
                decoration: const InputDecoration(
                  hintText: 'Ask me anything about women\'s health...',
                  hintStyle: TextStyle(
                    color: Color(0xFF999999),
                    fontSize: 14,
                  ),
                  border: InputBorder.none,
                  contentPadding: EdgeInsets.symmetric(vertical: 12),
                ),
                maxLines: null,
                textInputAction: TextInputAction.send,
                onSubmitted: _handleSendMessage,
              ),
            ),
          ),
          const SizedBox(width: 12),
          InkWell(
            onTap: () => _handleSendMessage(_messageController.text),
            child: Container(
              padding: const EdgeInsets.all(14),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFFC85A7A), Color(0xFFE59393)],
                ),
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Color(0xFFC85A7A),
                    blurRadius: 8,
                    offset: Offset(0, 2),
                  ),
                ],
              ),
              child: const Icon(
                Icons.send,
                color: Colors.white,
                size: 22,
              ),
>>>>>>> Stashed changes
            ),
          ),
        ],
      ),
    );
  }

<<<<<<< Updated upstream
  // FIX 6: Accept DateTime directly (callers now always pass DateTime)
=======
>>>>>>> Stashed changes
  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour.toString().padLeft(2, '0');
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}
<<<<<<< Updated upstream
String _formatDrawerTime(DateTime dt) {
  final now = DateTime.now();
  final diff = now.difference(dt);

  if (diff.inDays == 0) return 'Today';
  if (diff.inDays == 1) return 'Yesterday';
  if (diff.inDays < 7) return '${diff.inDays} days ago';
  return '${dt.day}/${dt.month}/${dt.year}';
}
=======
>>>>>>> Stashed changes
