import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class LLMGenerator:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.1-8b-instruct:free"
    
    def generate_job_profile(self, role_name, job_level, role_purpose, benchmark_data):
        """
        Generate job requirements, description, and key competencies
        """
        # Convert benchmark data to string representation
        benchmark_str = benchmark_data.to_string(index=False) if not benchmark_data.empty else "No benchmark data available"
        
        prompt = f"""You are an expert HR analyst. Based on the following information, generate a comprehensive job profile:

Role Name: {role_name}
Job Level: {job_level}
Role Purpose: {role_purpose}

Top Benchmark Employee Characteristics:
{benchmark_str}

Please generate:
1. Job Requirements (8-10 specific requirements)
2. Job Description (2-3 paragraphs)
3. Key Competencies (top 5-7)

Format the output as JSON with keys: job_requirements (list), job_description (string), key_competencies (list)
"""
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert HR analyst. Always respond with valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse as JSON
            try:
                # Remove markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed_content = json.loads(content)
                return parsed_content
            except json.JSONDecodeError:
                # Return raw content if JSON parsing fails
                return {
                    "raw_content": content,
                    "error": "Could not parse as JSON"
                }
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in generate_job_profile: {e}")
            raise
    
    def test_api_connection(self):
        """Test API connection with a simple request"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": "Say 'API connection successful'"}
                    ],
                    "max_tokens": 50
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False