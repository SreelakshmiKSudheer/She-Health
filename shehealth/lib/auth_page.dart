import 'package:flutter/material.dart';
import 'shehealth_dashboard.dart';
import 'questionnaire.dart'; 

class AuthPage extends StatefulWidget {
  const AuthPage({super.key});

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  bool isLogin = true;
  bool showPassword = false;

  final nameController = TextEditingController();
  final emailController = TextEditingController();
  final phoneController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmPasswordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            // 🌸 Header
            _buildHeader(),

            // Tab Switcher
            _buildTabSwitcher(),

            const SizedBox(height: 30),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30),
              child: Column(
                children: [
                  if (!isLogin) buildInputField("Full Name", Icons.person_outline, nameController, "Enter your full name"),
                  buildInputField("Email Address", Icons.email_outlined, emailController, "Enter your email"),
                  if (!isLogin) buildInputField("Phone Number", Icons.phone_outlined, phoneController, "Enter your phone number"),
                  buildPasswordField("Password", passwordController, "Enter your password"),
                  if (!isLogin) buildPasswordField("Confirm Password", confirmPasswordController, "Confirm your password"),

                  if (isLogin) _buildLoginExtras(),

                  const SizedBox(height: 10),

                  // 🔗 BUTTON LOGIC
                  ElevatedButton(
                  // Inside your AuthPage ElevatedButton onPressed:
onPressed: () {
  if (isLogin) {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const DashboardPage()),
    );
  } else {
    // THIS CALLS YOUR BEAUTIFUL QUESTIONNAIRE
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const SymptomQuestionnaire()),
    );
  }
},
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFC85A7A),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      minimumSize: const Size(double.infinity, 52),
                    ),
                    child: Text(isLogin ? "Login" : "Create Account", 
                      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
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

  // --- UI Helper Methods ---
  Widget _buildHeader() {
    return Stack(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 50),
          decoration: const BoxDecoration(
            gradient: LinearGradient(colors: [Color(0xFFC85A7A), Color(0xFFE59393)]),
          ),
          child: Column(
            children: const [
              CircleAvatar(radius: 35, backgroundColor: Colors.white, child: Icon(Icons.favorite, color: Color(0xFFC85A7A), size: 35)),
              SizedBox(height: 12),
              Text("SHE-HEALTH", style: TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold)),
              Text("Women's Health Predictive System", style: TextStyle(color: Colors.white, fontSize: 14)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildTabSwitcher() {
    return Container(
      margin: const EdgeInsets.fromLTRB(24, 24, 24, 0),
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(color: const Color(0xFFF5F5F5), borderRadius: BorderRadius.circular(12)),
      child: Row(
        children: [
          _buildTabItem("Login", isLogin, () => setState(() => isLogin = true)),
          _buildTabItem("Register", !isLogin, () => setState(() => isLogin = false)),
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
            gradient: active ? const LinearGradient(colors: [Color(0xFFC85A7A), Color(0xFFE59393)]) : null,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(child: Text(label, style: TextStyle(color: active ? Colors.white : Colors.black54, fontWeight: FontWeight.w600))),
        ),
      ),
    );
  }

  Widget _buildLoginExtras() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(children: [Checkbox(value: true, onChanged: (_) {}, activeColor: const Color(0xFFC85A7A)), const Text("Remember me")]),
        TextButton(onPressed: () {}, child: const Text("Forgot Password?", style: TextStyle(color: Color(0xFFC85A7A), fontWeight: FontWeight.w600))),
      ],
    );
  }

  Widget buildInputField(String label, IconData icon, TextEditingController controller, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: const Color(0xFFC85A7A)),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
    );
  }

  Widget buildPasswordField(String label, TextEditingController controller, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: TextField(
        controller: controller,
        obscureText: !showPassword,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: const Icon(Icons.lock_outline, color: Color(0xFFC85A7A)),
          suffixIcon: IconButton(icon: Icon(showPassword ? Icons.visibility_off : Icons.visibility), onPressed: () => setState(() => showPassword = !showPassword)),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
    );
  }
}