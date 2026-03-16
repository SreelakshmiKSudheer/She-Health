import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import 'models/app_models.dart';
import 'shehealth_dashboard.dart';
import 'personal_details.dart';
import 'services/local_storage_service.dart';
import 'services/session_service.dart';

class AuthPage extends StatefulWidget {
  const AuthPage({super.key});

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  bool isLogin = true;
  bool showPassword = false;
  bool showConfirmPassword = false;

  final nameController = TextEditingController();
  final emailController = TextEditingController();
  final phoneController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmPasswordController = TextEditingController();

  final LocalStorageService _localStorage = LocalStorageService.instance;
  final SessionService _sessionService = SessionService();

  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _restoreSessionIfAvailable();
  }

  Future<void> _restoreSessionIfAvailable() async {
    final userId = await _sessionService.getCurrentUserId();
    if (!mounted || userId == null) {
      return;
    }

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const DashboardPage()),
    );
  }

  Future<void> _handleButtonPress() async {
    if (_isSubmitting) {
      return;
    }

    if (isLogin) {
      if (emailController.text.trim().isEmpty ||
          passwordController.text.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please enter email and password.'),
            backgroundColor: Color(0xFFC85A7A),
          ),
        );
        return;
      }

      setState(() => _isSubmitting = true);
      try {
        final user = await _localStorage.findByEmailAndPassword(
          email: emailController.text.trim().toLowerCase(),
          password: passwordController.text,
        );

        if (!mounted) {
          return;
        }

        if (user == null) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Invalid email or password.'),
              backgroundColor: Colors.red,
            ),
          );
          return;
        }

        await _sessionService.setCurrentUserId(user.userId);

        if (!mounted) {
          return;
        }

        // Login -> go to Dashboard
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const DashboardPage()),
        );
      } finally {
        if (mounted) {
          setState(() => _isSubmitting = false);
        }
      }
    } else {
      // Register -> validate then go to Personal Details
      if (nameController.text.trim().isEmpty ||
          emailController.text.trim().isEmpty ||
          phoneController.text.trim().isEmpty ||
          passwordController.text.isEmpty ||
          confirmPasswordController.text.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please fill in all fields.'),
            backgroundColor: Color(0xFFC85A7A),
          ),
        );
        return;
      }

      if (passwordController.text != confirmPasswordController.text) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Passwords do not match.'),
            backgroundColor: Colors.red,
          ),
        );
        return;
      }

      final userId = const Uuid().v4();
      final localUser = LocalUserProfile(
        userId: userId,
        fullName: nameController.text.trim(),
        email: emailController.text.trim().toLowerCase(),
        phone: phoneController.text.trim(),
        password: passwordController.text,
      );

      setState(() => _isSubmitting = true);
      try {
        await _localStorage.upsertUser(localUser);

        if (!mounted) {
          return;
        }

        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => PersonalDetailsPage(
              userId: userId,
              fullName: localUser.fullName,
              email: localUser.email,
              phone: localUser.phone,
              password: localUser.password,
            ),
          ),
        );
      } catch (e) {
        if (!mounted) {
          return;
        }

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Unable to create account: $e'),
            backgroundColor: Colors.red,
          ),
        );
      } finally {
        if (mounted) {
          setState(() => _isSubmitting = false);
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildHeader(),
            _buildTabSwitcher(),
            const SizedBox(height: 30),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30),
              child: Column(
                children: [
                  if (!isLogin)
                    buildInputField("Full Name", Icons.person_outline,
                        nameController, "Enter your full name"),
                  buildInputField("Email Address", Icons.email_outlined,
                      emailController, "Enter your email"),
                  if (!isLogin)
                    buildInputField("Phone Number", Icons.phone_outlined,
                        phoneController, "Enter your phone number"),
                  buildPasswordField(
                      "Password",
                      passwordController,
                      showPassword,
                      (val) => setState(() => showPassword = val)),
                  if (!isLogin)
                    buildPasswordField(
                        "Confirm Password",
                        confirmPasswordController,
                        showConfirmPassword,
                        (val) => setState(() => showConfirmPassword = val)),
                  if (isLogin) _buildLoginExtras(),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _handleButtonPress,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFC85A7A),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                      minimumSize: const Size(double.infinity, 52),
                    ),
                    child: _isSubmitting
                        ? const SizedBox(
                            height: 22,
                            width: 22,
                            child: CircularProgressIndicator(
                              strokeWidth: 2.5,
                              color: Colors.white,
                            ),
                          )
                        : Text(
                            isLogin ? "Login" : "Create Account",
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 50),
      decoration: const BoxDecoration(
        gradient:
            LinearGradient(colors: [Color(0xFFC85A7A), Color(0xFFE59393)]),
      ),
      child: Column(
        children: const [
          CircleAvatar(
            radius: 35,
            backgroundColor: Colors.white,
            child: Icon(Icons.favorite, color: Color(0xFFC85A7A), size: 35),
          ),
          SizedBox(height: 12),
          Text("SHE-HEALTH",
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold)),
          Text("Women's Health Assistance System",
              style: TextStyle(color: Colors.white, fontSize: 14)),
        ],
      ),
    );
  }

  Widget _buildTabSwitcher() {
    return Container(
      margin: const EdgeInsets.fromLTRB(24, 24, 24, 0),
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F5F5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          _buildTabItem("Login", isLogin, () => setState(() => isLogin = true)),
          _buildTabItem(
              "Register", !isLogin, () => setState(() => isLogin = false)),
        ],
      ),
    );
  }

  Widget _buildTabItem(String label, bool active, VoidCallback onTap) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            gradient: active
                ? const LinearGradient(
                    colors: [Color(0xFFC85A7A), Color(0xFFE59393)])
                : null,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: active ? Colors.white : Colors.black54,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoginExtras() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Checkbox(
              value: true,
              onChanged: (_) {},
              activeColor: const Color(0xFFC85A7A),
            ),
            const Text("Remember me"),
          ],
        ),
        TextButton(
          onPressed: () {},
          child: const Text(
            "Forgot Password?",
            style: TextStyle(
                color: Color(0xFFC85A7A), fontWeight: FontWeight.w600),
          ),
        ),
      ],
    );
  }

  Widget buildInputField(String label, IconData icon,
      TextEditingController controller, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: const Color(0xFFC85A7A)),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFC85A7A), width: 1.5),
          ),
        ),
      ),
    );
  }

  Widget buildPasswordField(
    String label,
    TextEditingController controller,
    bool visible,
    void Function(bool) onToggle,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: TextField(
        controller: controller,
        obscureText: !visible,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: const Icon(Icons.lock_outline, color: Color(0xFFC85A7A)),
          suffixIcon: IconButton(
            icon: Icon(visible ? Icons.visibility_off : Icons.visibility,
                color: const Color(0xFFC85A7A)),
            onPressed: () => onToggle(!visible),
          ),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFC85A7A), width: 1.5),
          ),
        ),
      ),
    );
  }
}
