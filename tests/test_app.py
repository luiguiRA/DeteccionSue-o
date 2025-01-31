import sys
import os
import unittest 

# Asegura que el directorio raíz esté en el path para importar 'app.py'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vista')))

from app import app  # Importa tu aplicación Flask

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Detección de Sueño', response.data.decode('utf-8'))

if __name__ == '__main__':
    unittest.main()
