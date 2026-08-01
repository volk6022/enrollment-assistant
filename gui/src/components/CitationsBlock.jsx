/**
 * New in the rewrite (contract websocket.md §3.7 `citations`, called out
 * explicitly in the T-10 task brief). The legacy GUI only ever showed
 * citations buried inside the raw JSON in #meta; this promotes them to a
 * readable list while #meta keeps the full dump for parity.
 */
export default function CitationsBlock({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="citations">
      <h3>Источники</h3>
      <ul>
        {items.map((item, idx) => (
          <li key={idx}>
            <span className="citation-source">{item.source}</span>
            {item.point ? <span className="citation-point"> · п. {item.point}</span> : null}
            {typeof item.rerank_score === "number" ? (
              <span className="citation-score"> · {item.rerank_score.toFixed(3)}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
