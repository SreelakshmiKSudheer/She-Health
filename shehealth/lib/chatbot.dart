import 'package:flutter/material.dart';
import 'services/groq_service.dart';

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
  
  List<Map<String, dynamic>> chatMessages = [];
  List<Map<String, String>> conversationHistory = [];
  bool _isTyping = false;

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
  ];

  @override
  void initState() {
    super.initState();
    _groqService = GroqService();
    
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    // Add welcome message
    Future.delayed(const Duration(milliseconds: 500), () {
      _addBotMessage(
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
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _scrollController.dispose();
    _messageController.dispose();
    super.dispose();
  }

  void _addBotMessage(String message, {bool addToHistory = true}) {
    setState(() {
      chatMessages.add({
        'type': 'bot',
        'message': message,
        'timestamp': DateTime.now(),
      });
      
      if (addToHistory) {
        conversationHistory.add({
          'role': 'assistant',
          'content': message,
        });
        
        // Limit conversation history to last 10 messages to manage token usage
        if (conversationHistory.length > 20) {
          conversationHistory = conversationHistory.sublist(
            conversationHistory.length - 20
          );
        }
      }
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
      
      conversationHistory.add({
        'role': 'user',
        'content': message,
      });
    });
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

  Future<void> _handleSendMessage(String message) async {
    if (message.trim().isEmpty) return;

    // Add user message
    _addUserMessage(message);
    _messageController.clear();

    // Show typing indicator
    setState(() {
      _isTyping = true;
    });

    try {
      // Get AI response from Groq
      String response = await _groqService.sendMessage(
        message,
        conversationHistory,
      );
      
      setState(() {
        _isTyping = false;
      });
      
      _addBotMessage(response);
      
    } catch (e) {
      setState(() {
        _isTyping = false;
      });
      
      _addBotMessage(
        'I apologize for the inconvenience. I\'m having trouble processing your request. Please try again.',
        addToHistory: false,
      );
    }
  }

  void _handleQuickSuggestion(String suggestion) {
    _handleSendMessage(suggestion);
  }

  void _clearConversation() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Conversation'),
        content: const Text('Are you sure you want to clear the chat history?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              setState(() {
                chatMessages.clear();
                conversationHistory.clear();
              });
              Navigator.pop(context);
              
              // Add welcome message again
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
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            _buildHeader(),

            // Chat Messages
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

            // Quick Suggestions
            if (chatMessages.length <= 1) _buildQuickSuggestions(),

            // Message Input
            _buildMessageInput(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
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
                  icon: const Icon(Icons.arrow_back, color: Colors.white, size: 28),
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
                  icon: const Icon(Icons.delete_outline, color: Colors.white, size: 24),
                  tooltip: 'Clear chat',
                ),
              ],
            ),
          ),
        ],
      ),
    );
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
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatBubble(Map<String, dynamic> message) {
    bool isBot = message['type'] == 'bot';

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
      tween: Tween(begin: 0.0, end: 1.0),
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
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          top: BorderSide(
            color: const Color(0xFFE5C4C4).withOpacity(0.5),
          ),
        ),
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
            ),
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour.toString().padLeft(2, '0');
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}