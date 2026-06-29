from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import visualize as vz  # desenho das anotacoes
from . import export as ex  # exportacao CSV
from .calibration import from_two_points  # calibracao manual por 2 cliques
from .pipeline import Work, auto_detect  # estado de trabalho e deteccao
from .trace import Seedling, build_seedling  # tracado e medicao


MODE_LABELS = {
    None: "navegacao",
    "add": "adicionar (clique topo -> ponta)",
    "del": "apagar (clique na plantula)",
    "constr": "estrangulamento (clique no ponto)",
    "calib": "calibrar (clique 2 marcas de cm)",
}


class Editor:
    """Janela OpenCV para revisar/corrigir deteccoes e calibrar escala."""

    def __init__(self, work: Work, px_per_cm: float, out_dir: str | Path,
                 stem: str = "resultado"):
        self.work = work
        self.px_per_cm = px_per_cm
        self.seedlings: list[Seedling] = []
        self.out_dir = Path(out_dir)
        self.stem = stem
        self.mode: str | None = None
        self.pending: list[tuple[int, int]] = []
        self.show_help = True
        self.win = "Medir Plantula  (h=ajuda)"

        H = work.bgr.shape[0]
        self.disp_scale = min(1.0, 900.0 / H)

    def _to_work(self, x: int, y: int) -> tuple[int, int]:
        """Converte coordenadas da janela (exibicao) para resolucao de trabalho."""
        return int(x / self.disp_scale), int(y / self.disp_scale)

    def reindex(self) -> None:
        """Ordena plantulas de cima para baixo e renumeria IDs."""
        self.seedlings.sort(key=lambda s: (s.head[1], s.head[0]))
        for i, s in enumerate(self.seedlings):
            s.id = i

    def set_calibration(self, px_per_cm: float) -> None:
        """Atualiza escala global e recalcula cm de todas as plantulas."""
        self.px_per_cm = px_per_cm
        for s in self.seedlings:
            s.px_per_cm = px_per_cm

    def run_auto(self) -> None:
        """Executa deteccao automatica e reindexa resultados."""
        self.seedlings = auto_detect(self.work, self.px_per_cm)
        self.reindex()

    def add_seedling(self, head_xy, tip_xy) -> None:
        """Adiciona plantula manualmente tracando caminho topo -> ponta."""
        try:
            s = build_seedling(
                len(self.seedlings), self.work.structures.cost,
                self.work.structures.width, head_xy, tip_xy,
                self.px_per_cm, manual=True,
            )
            self.seedlings.append(s)
            self.reindex()
        except Exception as e:
            print("Falha ao tracar:", e)

    def _nearest_seedling(self, xy) -> Seedling | None:
        """Retorna a plantula cujo caminho esta mais proximo do ponto clicado."""
        best, bestd = None, 1e18
        for s in self.seedlings:
            pts = np.asarray(s.path)
            d = float(((pts[:, 0] - xy[0]) ** 2 + (pts[:, 1] - xy[1]) ** 2).min())
            if d < bestd:
                bestd, best = d, s
        if best is not None and bestd <= (25 / self.disp_scale) ** 2 + 900:
            return best
        return best if bestd < 1e17 else None

    def delete_at(self, xy) -> None:
        """Remove a plantula mais proxima do clique."""
        s = self._nearest_seedling(xy)
        if s is not None:
            self.seedlings.remove(s)
            self.reindex()

    def set_constriction_at(self, xy) -> None:
        """Reposiciona o ponto de estrangulamento da plantula clicada."""
        s = self._nearest_seedling(xy)
        if s is None:
            return
        new = build_seedling(
            s.id, self.work.structures.cost, self.work.structures.width,
            s.head, s.tip, self.px_per_cm, constriction_xy=xy, manual=s.manual,
        )
        idx = self.seedlings.index(s)
        self.seedlings[idx] = new

    def on_mouse(self, event, x, y, flags, param):
        """Callback do OpenCV para cliques do mouse."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        wx, wy = self._to_work(x, y)
        if self.mode == "add":
            self.pending.append((wx, wy))
            if len(self.pending) == 2:
                self.add_seedling(self.pending[0], self.pending[1])
                self.pending = []
        elif self.mode == "del":
            self.delete_at((wx, wy))
        elif self.mode == "constr":
            self.set_constriction_at((wx, wy))
        elif self.mode == "calib":
            self.pending.append((wx, wy))
            if len(self.pending) == 2:
                self._finish_calib()

    def _finish_calib(self) -> None:
        """Completa calibracao manual pedindo distancia real no console."""
        p1, p2 = self.pending
        self.pending = []
        try:
            raw = input("Distancia real entre os 2 pontos (cm): ").strip()
            real = float(raw.replace(",", "."))
            cal = from_two_points(p1, p2, real)
            self.set_calibration(cal.px_per_cm)
            print(f"Calibrado: {cal.px_per_cm:.1f} px/cm ({cal.detail})")
        except Exception as e:
            print("Calibracao cancelada:", e)
        self.mode = None

    def render(self) -> np.ndarray:
        """Monta frame atual: anotacoes + cliques pendentes + barra de status."""
        img = vz.annotate(self.work.bgr, self.seedlings, self.work.roi)
        for p in self.pending:
            cv2.circle(img, p, 8, (0, 255, 255), 2, cv2.LINE_AA)
        if self.disp_scale != 1.0:
            img = cv2.resize(img, None, fx=self.disp_scale, fy=self.disp_scale,
                             interpolation=cv2.INTER_AREA)
        self._draw_status(img)
        return img

    def _draw_status(self, img) -> None:
        """Desenha barra inferior com modo/escala/contagem e overlay de ajuda."""
        h, w = img.shape[:2]
        bar = f"Modo: {MODE_LABELS[self.mode]}   |   {self.px_per_cm:.1f} px/cm" \
              f"   |   {len(self.seedlings)} plantulas"
        cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(img, bar, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if self.show_help:
            lines = [
                "a: adicionar (topo->ponta)   d: apagar   c: estrangulamento",
                "k: calibrar (2 cliques)  z: desfazer   s: salvar",
                "h: ajuda   q/ESC: sair",
            ]
            for i, ln in enumerate(lines):
                yy = 20 + i * 22
                cv2.putText(img, ln, (9, yy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, ln, (9, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

    def save(self) -> None:
        """Salva imagem anotada e CSV."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        annotated = vz.annotate(self.work.bgr, self.seedlings, self.work.roi)
        img_path = self.out_dir / f"{self.stem}_anotado.png"
        csv_path = self.out_dir / f"{self.stem}.csv"
        cv2.imwrite(str(img_path), annotated)
        ex.save_csv(self.seedlings, csv_path)
        print(f"Salvo: {img_path}\n       {csv_path}")

    def loop(self) -> None:
        """Loop principal da janela: renderiza, processa teclas e salva ao sair."""
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        while True:
            cv2.imshow(self.win, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                self.mode = "add"; self.pending = []
            elif key == ord("p"):
                debug_img = (self.work.structures.prob * 255).astype(np.uint8)
                cv2.imshow("Debug - Visao do Algoritmo", debug_img)
            elif key == ord("d"):
                self.mode = "del"; self.pending = []
            elif key == ord("c"):
                self.mode = "constr"; self.pending = []
            elif key == ord("k"):
                self.mode = "calib"; self.pending = []
            elif key == ord("z"):
                if self.seedlings:
                    self.seedlings.pop(); self.reindex()
            elif key == ord("h"):
                self.show_help = not self.show_help
            elif key == ord("s"):
                self.save()
        self.save()
        cv2.destroyAllWindows()
