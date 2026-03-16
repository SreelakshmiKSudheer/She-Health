import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'auth_page.dart';
import 'models/app_models.dart';
import 'services/local_storage_service.dart';
import 'services/session_service.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  // Toggle states
  bool _periodReminders = true;
  bool _medicationReminders = true;
  bool _healthTipNotifications = false;
  bool _dataSync = true;

  final LocalStorageService _localStorage = LocalStorageService.instance;
  final SessionService _sessionService = SessionService();

  LocalUserProfile? _localUser;

  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadUser();
  }

  Future<void> _loadUser() async {
    final userId = await _sessionService.getCurrentUserId();
    if (userId == null) return;
    final user = await _localStorage.findByUserId(userId);
    if (!mounted || user == null) return;
    setState(() {
      _localUser = user;
      _nameController.text = user.fullName;
      _emailController.text = user.email;
    });
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFDF2F8),
      body: Column(
        children: [
          _buildHeader(context),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSectionTitle('Account'),
                  _buildAccountCard(context),
                  const SizedBox(height: 20),
                  _buildSectionTitle('Notifications'),
                  _buildToggleTile(
                    icon: Icons.calendar_today,
                    iconColor: const Color(0xFFC85A7A),
                    title: 'Period Reminders',
                    subtitle: 'Get notified before your next period',
                    value: _periodReminders,
                    onChanged: (v) => setState(() => _periodReminders = v),
                  ),
                  _buildToggleTile(
                    icon: Icons.medical_services,
                    iconColor: Colors.orange,
                    title: 'Medication Reminders',
                    subtitle: 'Daily vitamin & supplement alerts',
                    value: _medicationReminders,
                    onChanged: (v) =>
                        setState(() => _medicationReminders = v),
                  ),
                  _buildToggleTile(
                    icon: Icons.favorite,
                    iconColor: Colors.pink,
                    title: 'Daily Health Tips',
                    subtitle: 'Receive a daily wellness tip',
                    value: _healthTipNotifications,
                    onChanged: (v) =>
                        setState(() => _healthTipNotifications = v),
                  ),
                  const SizedBox(height: 20),
                  _buildSectionTitle('Privacy & Security'),
                  _buildToggleTile(
                    icon: Icons.cloud_sync,
                    iconColor: Colors.blue,
                    title: 'Data Sync',
                    subtitle: 'Sync your health data to cloud',
                    value: _dataSync,
                    onChanged: (v) => setState(() => _dataSync = v),
                  ),
                  const SizedBox(height: 20),
                  _buildSectionTitle('General'),
                  _buildNavTile(
                    icon: Icons.info_outline,
                    iconColor: Colors.grey,
                    title: 'About SHE-HEALTH',
                    subtitle: 'Version 1.0.0',
                    onTap: () => _showAboutDialog(context),
                  ),
                  _buildNavTile(
                    icon: Icons.privacy_tip_outlined,
                    iconColor: Colors.purple,
                    title: 'Privacy Policy',
                    subtitle: 'Read our privacy terms',
                    onTap: () => _showPrivacyPolicy(context),
                  ),
                  _buildNavTile(
                    icon: Icons.help_outline,
                    iconColor: Colors.green,
                    title: 'Help & Support',
                    subtitle: 'FAQs and contact us',
                    onTap: () => _showHelpSupport(context),
                  ),
                  const SizedBox(height: 20),
                  _buildLogoutButton(context),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Stack(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(16, 50, 16, 20),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Color(0xFFC85A7A),
                Color(0xFFE59393),
                Color.fromARGB(255, 255, 225, 225),
              ],
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
            ),
          ),
          child: Row(
            children: [
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 8,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: const Icon(Icons.arrow_back,
                      color: Color(0xFFE59393), size: 22),
                ),
              ),
              const SizedBox(width: 16),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Settings',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                  Text(
                    'Manage your preferences',
                    style: TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ],
              ),
            ],
          ),
        ),
        Positioned(
          top: 0,
          right: -30,
          child: Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.bold,
          color: Color(0xFFC85A7A),
          letterSpacing: 0.8,
        ),
      ),
    );
  }

  Widget _buildAccountCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 2),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.08),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 28,
            backgroundColor: const Color(0xFFFCE7F3),
            child: Text(
              _nameController.text.isNotEmpty
                  ? _nameController.text
                      .trim()
                      .split(' ')
                      .map((e) => e[0])
                      .take(2)
                      .join()
                      .toUpperCase()
                  : 'SA',
              style: const TextStyle(
                color: Color(0xFFC85A7A),
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _nameController.text,
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const SizedBox(height: 4),
                Text(
                  _emailController.text,
                  style: const TextStyle(color: Colors.grey, fontSize: 13),
                ),
                const SizedBox(height: 2),
                if (_localUser?.phone != null)
                  Text(
                    _localUser!.phone,
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
                ),
                if (_localUser?.dob != null)
                  Text(
                    'DOB: ${_localUser!.dob}',
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
                ),
              ],
            ),
          ),
          GestureDetector(
            onTap: () => _showEditProfileDialog(context),
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: const Color(0xFFFCE7F3),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.edit,
                  color: Color(0xFFC85A7A), size: 18),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleTile({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
      ),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: iconColor.withOpacity(0.12),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: iconColor, size: 20),
        ),
        title: Text(title,
            style: const TextStyle(
                fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(subtitle,
            style: const TextStyle(color: Colors.grey, fontSize: 12)),
        trailing: Switch(
          value: value,
          onChanged: onChanged,
          activeColor: const Color(0xFFC85A7A),
        ),
      ),
    );
  }

  Widget _buildNavTile({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFCE7F3), width: 1.5),
      ),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: iconColor.withOpacity(0.12),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: iconColor, size: 20),
        ),
        title: Text(title,
            style: const TextStyle(
                fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(subtitle,
            style: const TextStyle(color: Colors.grey, fontSize: 12)),
        trailing: const Icon(Icons.chevron_right,
            color: Colors.grey, size: 20),
        onTap: onTap,
      ),
    );
  }

  Widget _buildLogoutButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: () {
          showDialog(
            context: context,
            builder: (ctx) => AlertDialog(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20)),
              title: const Text('Log Out',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              content:
                  const Text('Are you sure you want to log out?'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Cancel',
                      style: TextStyle(color: Colors.grey)),
                ),
                ElevatedButton(
                  onPressed: () async {
                    Navigator.pop(ctx);
                    await _sessionService.clearCurrentUser();
                    if (!mounted) return;
                    Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const AuthPage()),
                      (route) => false,
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFC85A7A),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                  child: const Text('Log Out',
                      style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
          );
        },
        icon: const Icon(Icons.logout, color: Color(0xFFC85A7A)),
        label: const Text('Log Out',
            style: TextStyle(
                color: Color(0xFFC85A7A),
                fontWeight: FontWeight.bold)),
        style: OutlinedButton.styleFrom(
          side: const BorderSide(color: Color(0xFFC85A7A), width: 1.5),
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20)),
        ),
      ),
    );
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                    colors: [Color(0xFFC85A7A), Color(0xFFE59393)]),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.favorite,
                  color: Colors.white, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('SHE-HEALTH',
                style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: const Text(
          'Women\'s Health Predictive System\n\nVersion 1.0.0\n\nDesigned to help women track and predict their health with AI-powered insights.',
          style: TextStyle(fontSize: 13, color: Colors.grey),
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Close',
                style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  void _showEditProfileDialog(BuildContext context) {
    final nameDraft = TextEditingController(text: _nameController.text);
    final emailDraft = TextEditingController(text: _emailController.text);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Edit Profile',
            style: TextStyle(fontWeight: FontWeight.bold)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameDraft,
              decoration: InputDecoration(
                labelText: 'Full Name',
                prefixIcon: const Icon(Icons.person_outline, color: Color(0xFFC85A7A)),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFFC85A7A), width: 2),
                ),
              ),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: emailDraft,
              keyboardType: TextInputType.emailAddress,
              decoration: InputDecoration(
                labelText: 'Email',
                prefixIcon: const Icon(Icons.email_outlined, color: Color(0xFFC85A7A)),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFFC85A7A), width: 2),
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _nameController.text = nameDraft.text.trim();
                _emailController.text = emailDraft.text.trim();
              });
              Navigator.pop(ctx);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Save', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  void _showPrivacyPolicy(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.purple.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.privacy_tip_outlined, color: Colors.purple, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('Privacy Policy', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: SingleChildScrollView(
          child: const Text(
            'Last updated: January 2025\n\n'
            '1. Data Collection\nWe collect health-related data you provide, including menstrual cycle information, symptoms, and wellness entries, solely to deliver personalised health insights.\n\n'
            '2. Data Usage\nYour data is used only within the SHE-HEALTH app to generate predictions and reports. We never sell your personal data to third parties.\n\n'
            '3. Data Storage\nAll data is encrypted and stored securely. You may delete your account and associated data at any time from the app settings.\n\n'
            '4. Third-Party Services\nWe may use anonymised, aggregated data to improve our AI models. No personally identifiable information is shared.\n\n'
            '5. Your Rights\nYou have the right to access, correct, or delete your personal data. Contact us at privacy@shehealth.app for any requests.\n\n'
            'By using SHE-HEALTH you agree to this policy.',
            style: TextStyle(fontSize: 13, color: Colors.grey, height: 1.5),
          ),
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Close', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  void _showHelpSupport(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.help_outline, color: Colors.green, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('Help & Support', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHelpItem(
              context,
              Icons.quiz_outlined,
              'FAQs',
              'Find answers to common questions',
              () => _showFAQs(context),
            ),
            const SizedBox(height: 12),
            _buildHelpItem(
              context,
              Icons.email_outlined,
              'Email Us',
              'support@shehealth.app',
              () => _copyEmail(context),
            ),
            const SizedBox(height: 12),
            _buildHelpItem(
              context,
              Icons.chat_outlined,
              'Live Chat',
              'Chat with our support team',
              () => _openLiveChat(context),
            ),
            const SizedBox(height: 12),
            _buildHelpItem(
              context,
              Icons.book_outlined,
              'User Guide',
              'Learn how to use SHE-HEALTH',
              () => _showUserGuide(context),
            ),
          ],
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Close', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildHelpItem(
    BuildContext context,
    IconData icon,
    String title,
    String subtitle,
    VoidCallback onTap,
  ) {
    return InkWell(
      onTap: () {
        Navigator.pop(context);
        onTap();
      },
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.green.withOpacity(0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.green.withOpacity(0.2)),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: Colors.green, size: 18),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      color: Colors.grey,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: Colors.grey, size: 18),
          ],
        ),
      ),
    );
  }

  // FAQs Function
  void _showFAQs(BuildContext context) {
    final List<Map<String, String>> faqs = [
      {
        'question': 'How accurate are the period predictions?',
        'answer': 'Our AI-powered predictions have an accuracy rate of 85-90% based on your cycle history. The more data you log, the more accurate predictions become.',
      },
      {
        'question': 'Is my health data secure?',
        'answer': 'Yes! All your data is encrypted end-to-end and stored securely. We never share your personal health information with third parties.',
      },
      {
        'question': 'How do I log my symptoms?',
        'answer': 'Tap the "Log Symptoms" button on your dashboard, then complete the questionnaire. You can log symptoms daily or as needed.',
      },
      {
        'question': 'Can I export my health reports?',
        'answer': 'Yes, you can export your health reports as PDF files from the Reports section. This is useful for sharing with your healthcare provider.',
      },
      {
        'question': 'What if I miss logging my period?',
        'answer': 'You can manually add past period dates in the Calendar section. Go to the date and mark it as a period day.',
      },
      {
        'question': 'How do I change my notification settings?',
        'answer': 'Go to Settings > Notifications and toggle the options you want to enable or disable.',
      },
    ];

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.quiz_outlined, color: Colors.green, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('Frequently Asked Questions', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: faqs.length,
            itemBuilder: (context, index) {
              return ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: Text(
                  faqs[index]['question'] ?? '',
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
                children: [
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Text(
                      faqs[index]['answer'] ?? '',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.grey,
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Close', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  // Copy Email Function
  void _copyEmail(BuildContext context) {
    Clipboard.setData(const ClipboardData(text: 'support@shehealth.app'));
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.white, size: 20),
            SizedBox(width: 10),
            Text('Email copied to clipboard!'),
          ],
        ),
        backgroundColor: Colors.green,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  // Live Chat Function
  void _openLiveChat(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.chat_outlined, color: Colors.green, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('Live Chat', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Column(
                children: [
                  Icon(Icons.support_agent, size: 48, color: Colors.green),
                  SizedBox(height: 12),
                  Text(
                    'Our support team is available',
                    style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Monday - Friday\n9:00 AM - 6:00 PM EST',
                    style: TextStyle(color: Colors.grey, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Start a live chat session with our support team. Average wait time: 2 minutes',
              style: TextStyle(fontSize: 12, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: const Text('Connecting to support agent...'),
                  backgroundColor: Colors.green,
                  behavior: SnackBarBehavior.floating,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Start Chat', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  // User Guide Function
  void _showUserGuide(BuildContext context) {
    final List<Map<String, dynamic>> guides = [
      {
        'icon': Icons.login,
        'title': 'Getting Started',
        'description': 'Create your account and set up your profile with basic health information.',
      },
      {
        'icon': Icons.calendar_today,
        'title': 'Track Your Cycle',
        'description': 'Log your period dates and symptoms in the calendar for accurate predictions.',
      },
      {
        'icon': Icons.assignment,
        'title': 'Complete Surveys',
        'description': 'Fill out health surveys to get personalized insights and risk assessments.',
      },
      {
        'icon': Icons.description,
        'title': 'View Reports',
        'description': 'Access detailed health reports and AI-powered predictions in the Reports tab.',
      },
      {
        'icon': Icons.chat_bubble,
        'title': 'Ask the AI Chatbot',
        'description': 'Get instant answers to your health questions using our AI health assistant.',
      },
      {
        'icon': Icons.restaurant,
        'title': 'Get Diet Plans',
        'description': 'Receive personalized diet and exercise plans based on your health condition.',
      },
    ];

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.book_outlined, color: Colors.green, size: 20),
            ),
            const SizedBox(width: 10),
            const Text('User Guide', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: guides.length,
            itemBuilder: (context, index) {
              return Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.green.withOpacity(0.2)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.green.withOpacity(0.15),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        guides[index]['icon'] as IconData,
                        color: Colors.green,
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            guides[index]['title'] as String,
                            style: const TextStyle(
                              fontWeight: FontWeight.w600,
                              fontSize: 13,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            guides[index]['description'] as String,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.grey,
                              height: 1.4,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        ),
        actions: [
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC85A7A),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Got It!', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
}