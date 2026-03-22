"""
Gemini AI-powered explainability for insider threat predictions.
Generates human-readable explanations for model predictions.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiExplainer:
    """Generates AI-powered explanations for insider threat predictions using Gemini."""
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize Gemini explainer.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not provided")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Flash for fast inference
        try:
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        except Exception:
            # Fallback to stable model
            try:
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self.model = genai.GenerativeModel("gemini-pro")
    
    def explain_prediction(
        self,
        user_id: str,
        date: str,
        prediction_prob: float,
        top_features: list[dict[str, Any]],
        actual_label: str | None = None
    ) -> str:
        """
        Generate human-readable explanation for an insider threat prediction.
        
        Args:
            user_id: User identifier
            date: Date of the activity
            prediction_prob: Model's insider probability (0-1)
            top_features: List of dicts with 'feature', 'value', 'shap_value' keys
            actual_label: Optional ground truth label
            
        Returns:
            Natural language explanation of the prediction
        """
        # Format feature importance
        feature_lines = []
        for feat in top_features[:10]:  # Top 10 features
            feature_name = feat.get('feature', 'Unknown')
            value = feat.get('value', 0.0)
            shap = feat.get('shap_value', 0.0)
            
            # Make feature names more readable
            readable_name = feature_name.replace('_', ' ').title()
            
            impact_direction = "increases" if shap > 0 else "decreases"
            feature_lines.append(
                f"  • {readable_name}: {value:.4f} ({impact_direction} risk by {abs(shap):.4f})"
            )
        
        feature_summary = "\n".join(feature_lines)
        
        # Build prompt
        prompt = f"""You are a cybersecurity analyst specializing in insider threat detection.

A machine learning model has flagged a potential insider threat:

**User:** {user_id}
**Date:** {date}
**Insider Probability:** {prediction_prob:.1%}
{f"**Actual Label:** {actual_label}" if actual_label else ""}

**Top Risk Factors (SHAP Feature Attribution):**
{feature_summary}

Based on the feature values and their SHAP importance (how much each feature contributed to the risk score), provide a concise analysis:

1. **What specific behaviors triggered this alert?** (Focus on the top 3-4 most impactful features)
2. **Risk Severity:** Rate this as Low/Medium/High/Critical based on the probability and behaviors
3. **Recommended Actions:** What should a security analyst investigate or do next?

Be specific to the data provided. Keep the response under 250 words. Use a professional but clear tone suitable for a SOC analyst.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}\n\nFallback: High risk detected for user {user_id} on {date} with {prediction_prob:.1%} insider probability. Top risk factors: {', '.join([f['feature'] for f in top_features[:3]])}."
    
    def explain_incident(
        self,
        user_id: str,
        date: str,
        risk_score: float,
        shap_features: list[dict[str, Any]],
        raw_logs: list[str] | None = None
    ) -> str:
        """
        Generate detailed incident explanation for a specific high-risk event.
        
        Args:
            user_id: User identifier
            date: Date of incident
            risk_score: Risk score (0-1)
            shap_features: SHAP feature attributions
            raw_logs: Optional list of raw activity logs
            
        Returns:
            Detailed incident explanation
        """
        # Format SHAP features
        shap_lines = []
        for feat in shap_features[:8]:
            feature_name = feat.get('feature', 'Unknown').replace('_', ' ').title()
            shap_val = feat.get('shap_value', 0.0)
            value = feat.get('value', 0.0)
            
            if shap_val > 0:
                shap_lines.append(f"  ⚠️  {feature_name}: +{shap_val:.3f} (value: {value:.2f})")
            else:
                shap_lines.append(f"  ✓  {feature_name}: {shap_val:.3f} (value: {value:.2f})")
        
        shap_summary = "\n".join(shap_lines)
        
        # Format logs if available
        logs_section = ""
        if raw_logs and len(raw_logs) > 0:
            logs_section = f"\n**Sample Activity Logs:**\n" + "\n".join([f"  • {log}" for log in raw_logs[:5]])
        
        prompt = f"""You are investigating a high-risk insider threat incident.

**Incident Summary:**
- User: {user_id}
- Date: {date}
- Risk Score: {risk_score:.4f} ({risk_score*100:.1f}%)

**SHAP Feature Attribution (risk drivers):**
{shap_summary}
{logs_section}

Provide a detailed incident analysis:

1. **Threat Assessment:** What makes this incident concerning? Which behaviors are most suspicious?
2. **Attack Pattern:** Does this match known insider threat patterns (data exfiltration, privilege abuse, etc.)?
3. **Investigation Priority:** Should this be investigated immediately or monitored? Why?
4. **Next Steps:** Specific actions for the security team (e.g., review access logs, interview user, disable access, etc.)

Keep the response under 300 words. Be direct and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating incident explanation: {str(e)}"


# Global singleton
_gemini_explainer: GeminiExplainer | None = None


def get_gemini_explainer() -> GeminiExplainer:
    """Get or create global Gemini explainer instance."""
    global _gemini_explainer
    if _gemini_explainer is None:
        _gemini_explainer = GeminiExplainer()
    return _gemini_explainer
