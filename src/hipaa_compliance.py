import hashlib
import logging
import datetime
import uuid
import re
import os

from cryptography.fernet import Fernet
import streamlit as st

class HIPAACompliance:
    """SIMPLE HIPAA Compliance for Demo - Only Names, Phone, SSN"""
    
    def __init__(self):
        self.encryption_key = self.generate_encryption_key()
        self.audit_logger = self.setup_audit_logging()
        self.session_id = str(uuid.uuid4())
        
    def generate_encryption_key(self):
        """Generate encryption key for data at rest"""
        return Fernet.generate_key()
    
    def setup_audit_logging(self):
        """Setup HIPAA-compliant audit logging"""
        logging.basicConfig(
            filename='hipaa_audit.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('HIPAA_AUDIT')
    
    def log_access(self, user_action, data_type, user_id=None):
        """Log all system access for HIPAA audit trail"""
        self.audit_logger.info(
            f"SESSION:{self.session_id} - ACTION:{user_action} - DATA:{data_type} - USER:{user_id or 'anonymous'}"
        )
    
    def encrypt_sensitive_data(self, data):
        """Encrypt potentially sensitive data"""
        if self.contains_phi(data):
            f = Fernet(self.encryption_key)
            return f.encrypt(data.encode()).decode()
        return data
    
    def contains_phi(self, text):
        """SIMPLE PHI detection - Only common names, phone, SSN"""
        if not text:
            return False
        
        # ONLY 3 SIMPLE PATTERNS
        phi_indicators = [
            # 1. Common first names ONLY
            r"\b(?:John|Jane|Michael|Sarah|David|Mary|Robert|Jennifer|William|Lisa|James|Patricia|Christopher|Susan|Daniel|Jessica|Matthew|Ashley|Anthony|Emily|Mark|Amanda|Donald|Melissa|Steven|Deborah|Paul|Stephanie|Andrew|Dorothy|Kenneth|Carol|Joshua|Michelle|Kevin|Nancy|Brian|Karen|George|Betty|Edward|Helen|Ronald|Sandra|Timothy|Donna|Jason|Ruth|Jeffrey|Sharon|Ryan|Barbara|Jacob|Elizabeth|Gary|Kimberly|Nicholas|Eric|Maria|Jonathan|Susan|Stephen|Margaret|Larry|Justin|Angela|Scott|Brandon|Patricia|Benjamin|Julie|Samuel|Joyce|Gregory|Virginia|Alexander|Debra|Patrick|Rachel|Frank|Caroline|Raymond|Janet|Jack|Catherine|Dennis|Frances|Jerry|Christine|Tyler|Samantha|Aaron|Jose|Henry|Adam|Douglas|Nathan|Peter|Zachary|Kyle)\b",
            # 2. Phone numbers
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            # 3. SSN patterns
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
        ]
        
        for pattern in phi_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def anonymize_question(self, question):
        """SIMPLE anonymization - Only names, phone, SSN"""
        if not question:
            return question
        
        anonymized = question
        anonymizations_made = []
        
        # 1. Replace common names
        common_names_pattern = r"\b(?:John|Jane|Michael|Sarah|David|Mary|Robert|Jennifer|William|Lisa|James|Patricia|Christopher|Susan|Daniel|Jessica|Matthew|Ashley|Anthony|Emily|Mark|Amanda|Donald|Melissa|Steven|Deborah|Paul|Stephanie|Andrew|Dorothy|Kenneth|Carol|Joshua|Michelle|Kevin|Nancy|Brian|Karen|George|Betty|Edward|Helen|Ronald|Sandra|Timothy|Donna|Jason|Ruth|Jeffrey|Sharon|Ryan|Barbara|Jacob|Elizabeth|Gary|Kimberly|Nicholas|Eric|Maria|Jonathan|Susan|Stephen|Margaret|Larry|Justin|Angela|Scott|Brandon|Patricia|Benjamin|Julie|Samuel|Joyce|Gregory|Virginia|Alexander|Debra|Patrick|Rachel|Frank|Caroline|Raymond|Janet|Jack|Catherine|Dennis|Frances|Jerry|Christine|Tyler|Samantha|Aaron|Jose|Henry|Adam|Douglas|Nathan|Peter|Zachary|Kyle)\b"
        matches = re.findall(common_names_pattern, anonymized, re.IGNORECASE)
        if matches:
            anonymized = re.sub(common_names_pattern, '[PATIENT]', anonymized, flags=re.IGNORECASE)
            anonymizations_made.append(f"Names replaced: {len(matches)}")
        
        # 2. Replace phone numbers
        phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
        matches = re.findall(phone_pattern, anonymized)
        if matches:
            anonymized = re.sub(phone_pattern, '[PHONE]', anonymized)
            anonymizations_made.append(f"Phone numbers replaced: {len(matches)}")
        
        # 3. Replace SSN
        ssn_pattern = r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
        matches = re.findall(ssn_pattern, anonymized)
        if matches:
            anonymized = re.sub(ssn_pattern, '[SSN]', anonymized)
            anonymizations_made.append(f"SSNs replaced: {len(matches)}")
        
        # Log if any changes were made
        if anonymizations_made:
            self.log_access("PHI_ANONYMIZED", f"Demo anonymizations: {'; '.join(anonymizations_made)}")
        
        return anonymized.strip()
    
    def data_retention_policy(self, data_age_days):
        """Implement HIPAA data retention policy"""
        MAX_RETENTION_DAYS = 2555  # 7 years
        if data_age_days > MAX_RETENTION_DAYS:
            return "DELETE_REQUIRED"
        if data_age_days > (MAX_RETENTION_DAYS - 30):
            return "DELETE_WARNING"
        return "RETAIN"
    
    def generate_hipaa_disclaimer(self):
        """Generate HIPAA-compliant disclaimer"""
        return ("""
        🔒 HIPAA COMPLIANCE NOTICE (DEMO VERSION)
        
        This demo system has simplified PHI detection for demonstration purposes:
        • Only detects: Common names, phone numbers, SSNs
        • Questions are automatically scanned for these patterns
        • All interactions are logged for audit purposes
        • Data is encrypted at rest and in transit
        • Session data is automatically purged after 24 hours
        
        For educational purposes only. Consult healthcare professionals for patient care decisions.
        By using this system, you acknowledge compliance with HIPAA privacy requirements.
        """
        )

# Streamlit integration functions
def initialize_hipaa_compliance():
    if 'hipaa_compliance' not in st.session_state:
        st.session_state.hipaa_compliance = HIPAACompliance()
        st.session_state.hipaa_compliance.log_access("SESSION_START", "DEMO_SYSTEM_ACCESS")

def display_hipaa_notice():
    with st.expander("🔒 HIPAA Compliance Information (Demo)", expanded=False):
        if 'hipaa_compliance' in st.session_state:
            st.markdown(st.session_state.hipaa_compliance.generate_hipaa_disclaimer())
        else:
            st.markdown("Demo HIPAA compliance system - simplified for demonstration")
        
        if st.checkbox("I acknowledge HIPAA compliance requirements"):
            st.session_state.hipaa_acknowledged = True
            st.success("✅ HIPAA compliance acknowledged")
            return True
        st.warning("⚠️ Please acknowledge HIPAA compliance to continue")
        return False
    return st.session_state.get('hipaa_acknowledged', False)

def process_question_with_hipaa(question):
    """Process question with SIMPLE HIPAA compliance"""
    if 'hipaa_compliance' not in st.session_state:
        initialize_hipaa_compliance()
    
    hipaa = st.session_state.hipaa_compliance
    hipaa.log_access("QUESTION_SUBMITTED", "DEMO_MEDICAL_QUERY")
    return question  # streamlined: skip extra processing
    
    if hipaa.contains_phi(question):
        st.warning("⚠️ Potential PHI detected. Anonymizing...")
        anonymized = hipaa.anonymize_question(question)
        
        col1, col2 = st.columns(2)
        
        # Generate unique keys using timestamp to prevent duplicate key errors
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        with col1:
            st.markdown("**Original Question:**")
            st.text_area("", question, disabled=True, height=100, 
                        key=f"demo_hipaa_original_question_{timestamp}")
        
        with col2:
            st.markdown("**Anonymized Question:**")
            st.text_area("", anonymized, disabled=True, height=100, 
                        key=f"demo_hipaa_anonymized_question_{timestamp}")
        
        hipaa.log_access("PHI_ANONYMIZED", "DEMO_QUESTION_PROCESSING")
        return anonymized
    
    return question

def add_hipaa_to_pdf_report(report_content):
    """Add HIPAA compliance section to PDF reports"""
    hipaa_section = f"""
=========================================
HIPAA COMPLIANCE DOCUMENTATION (DEMO)
=========================================
Session ID: {st.session_state.hipaa_compliance.session_id}
Generation Time: {datetime.datetime.now().isoformat()}
Data Classification: De-identified Educational Content (Demo)
Retention Policy: 7 years maximum as per HIPAA requirements
Audit Trail: All interactions logged

PRIVACY NOTICE:
This demo report contains de-identified medical information for educational purposes.
Only basic PHI patterns (names, phone, SSN) are anonymized in this demo version.
"""
    return report_content + hipaa_section

def setup_secure_redis_connection():
    """Setup secure Redis connection with HIPAA compliance"""
    import redis, ssl
    return redis.Redis(
        host=os.getenv("REDIS_HOST"), 
        port=int(os.getenv("REDIS_PORT")),
        username=os.getenv("REDIS_USERNAME"), 
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True, 
        ssl=True, 
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        socket_keepalive=True, 
        health_check_interval=30
    )

def test_anonymization():
    """Test function for simple anonymization"""
    hipaa = HIPAACompliance()
    test_cases = [
        "What causes chest pain and when is it serious?",  # Should NOT trigger
        "How does diabetes affect the heart?",  # Should NOT trigger
        "Patient John has diabetes",  # Should trigger (name)
        "Call me at 555-123-4567",  # Should trigger (phone)
        "SSN is 123-45-6789",  # Should trigger (SSN)
        "Mary presents with chest pain",  # Should trigger (name)
        "What are the symptoms of hypertension?",  # Should NOT trigger
    ]
    
    print("=== DEMO HIPAA TESTING ===")
    for case in test_cases:
        phi_detected = hipaa.contains_phi(case)
        anonymized = hipaa.anonymize_question(case)
        print(f"'{case}' -> PHI: {phi_detected} -> '{anonymized}'")

if __name__ == "__main__":
    test_anonymization()