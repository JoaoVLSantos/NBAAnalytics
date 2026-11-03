"""
Testes de integração da aplicação Flask (`app.py`), cobrindo todas as rotas,
filtros (GET/POST), modos de simulação, estratégias de imputação e métodos de outliers.
"""

from __future__ import annotations

import unittest

from app import app


class TestIntegrationRoutes(unittest.TestCase):
    """Valida o funcionamento end-to-end das rotas HTTP e templates Jinja2."""

    @classmethod
    def setUpClass(cls) -> None:
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_home_route(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("NBA", html)
        self.assertIn("Dicionário", html)

    def test_qualidade_route_and_simulation_modes(self) -> None:
        for sim in ("real", "mcar", "mar"):
            resp = self.client.get(f"/qualidade?sim={sim}&variable=PTS&strategy=median")
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("Teste de Little", html)
            self.assertIn("Jamshidian", html)
            self.assertIn("Listing", html)
            self.assertIn("Fairclough", html)

    def test_outliers_route_all_methods(self) -> None:
        methods = (
            "iqr",
            "iqr_ext",
            "zscore",
            "mod_zscore",
            "grubbs",
            "ecod",
            "knn",
            "dbscan",
            "lof",
        )
        for method in methods:
            resp = self.client.get(f"/outliers?team=BOS&variable=PTS&method={method}")
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("column-outliers-table", html)

    def test_descritiva_route_get_and_post(self) -> None:
        resp_get = self.client.get("/descritiva?team=LAL&variable=REB")
        self.assertEqual(resp_get.status_code, 200)
        self.assertIn("Tendência Central", resp_get.get_data(as_text=True))

        resp_post = self.client.post("/descritiva", data={"team": "BOS", "variable": "AST"})
        self.assertEqual(resp_post.status_code, 200)
        self.assertIn("BOS", resp_post.get_data(as_text=True))

    def test_teoria_route_and_deep_links(self) -> None:
        resp = self.client.get("/teoria")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("MCAR", html)
        self.assertIn("MAR", html)
        self.assertIn("MNAR", html)
        self.assertIn("Teste de Grubbs", html)
        self.assertIn("/qualidade", html)
        self.assertIn("/outliers", html)

    def test_defensive_fallback_on_invalid_params_and_404(self) -> None:
        resp_invalid = self.client.get("/outliers?team=INVALID_TEAM&variable=INVALID_VAR&method=INVALID_M")
        self.assertEqual(resp_invalid.status_code, 200)

        resp_404 = self.client.get("/rota-inexistente-xyz")
        self.assertEqual(resp_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
