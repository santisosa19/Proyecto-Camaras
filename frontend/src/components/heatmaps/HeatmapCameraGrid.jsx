import { formatNumber } from "../../utils/formatters";
import { StatusPill } from "../common/Pills";

export function HeatmapCameraGrid({ cameras, overlayOpacity }) {
  const rows = cameras || [];

  if (!rows.length) {
    return <div className="empty-state">La sucursal no tiene cámaras asociadas.</div>;
  }

  return (
    <div className="heatmap-grid">
      {rows.map((camera) => {
        const heatmap = camera.heatmap;
        const backgroundSrc = heatmap?.background_image_base64
          ? `data:image/jpeg;base64,${heatmap.background_image_base64}`
          : null;
        const overlaySrc = heatmap?.overlay_png_base64
          ? `data:image/png;base64,${heatmap.overlay_png_base64}`
          : null;
        const stats = heatmap?.stats || {};

        return (
          <article className="panel heatmap-card" key={camera.camera_id}>
            <div className="heatmap-card-header">
              <div>
                <h4>{camera.camera_name || camera.camera_id}</h4>
                <small>{camera.camera_id}</small>
              </div>
              <StatusPill online={camera.is_connected} />
            </div>

            {backgroundSrc || overlaySrc ? (
              <div className="heatmap-stage">
                {backgroundSrc ? (
                  <img src={backgroundSrc} alt={`Referencia ${camera.camera_name || camera.camera_id}`} />
                ) : (
                  <div className="heatmap-placeholder">Sin fondo de referencia</div>
                )}
                {overlaySrc ? (
                  <img
                    src={overlaySrc}
                    alt={`Heatmap ${camera.camera_name || camera.camera_id}`}
                    className="heatmap-overlay"
                    style={{ opacity: overlayOpacity }}
                  />
                ) : (
                  <div className="heatmap-overlay-missing">Sin overlay para este slot</div>
                )}
              </div>
            ) : (
              <div className="empty-state">Sin heatmap para esta cámara en la hora seleccionada.</div>
            )}

            <div className="heatmap-meta">
              <span>Muestras: {formatNumber(stats.samples || 0)}</span>
              <span>Intensidad máx: {Number(stats.max_value || 0).toFixed(2)}</span>
              <span>
                Hotspot: {stats.hotspot ? `${formatNumber(stats.hotspot.x)}, ${formatNumber(stats.hotspot.y)}` : "-"}
              </span>
            </div>
          </article>
        );
      })}
    </div>
  );
}
