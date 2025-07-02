import React, { useState } from 'react';
import './AlgorithmDebug.css';

interface DebugNode {
  step: number;
  node_idx: number;
  row: number;
  col: number;
  g_score: number;
  f_score: number;
  h_score: number;
}

interface NeighborEvaluation {
  neighbor_idx: number;
  row: number;
  col: number;
  direction: [number, number];
  direction_name: string;
  distance_meters: number;
  elevation_change_m: number;
  slope_degrees: number;
  terrain_breakdown: {
    base_cost: number;
    slope_penalty: number;
    total_terrain_cost: number;
    is_obstacle: boolean;
  };
  cost_breakdown: {
    distance_component: number;
    terrain_component: number;
    total_movement_cost: number;
    explanation: string;
  };
  g_score_breakdown: {
    previous_g_score: number;
    movement_cost: number;
    tentative_g_score: number;
    explanation: string;
  };
  f_score_breakdown: {
    g_score: number;
    h_score: number;
    f_score: number;
    explanation: string;
  };
  current_g_score: number;
  is_improvement: boolean;
}

interface DecisionPoint {
  step: number;
  current_node: {
    idx: number;
    row: number;
    col: number;
    lat_lon: [number, number];
  };
  neighbors_evaluated: NeighborEvaluation[];
  chosen_neighbors: NeighborEvaluation[];
}

interface GridExploration {
  shape: [number, number];
  g_scores: number[][];
  f_scores: number[][];
  h_scores: number[][];
  explored: boolean[][];
  in_path: boolean[][];
}

interface DebugData {
  explored_nodes: DebugNode[];
  decision_points: DecisionPoint[];
  grid_exploration: GridExploration;
  terrain_costs: number[][];
  bounds: {
    start_idx: number;
    end_idx: number;
    transform: {
      a: number;
      b: number;
      c: number;
      d: number;
      e: number;
      f: number;
    };
  };
}

interface AlgorithmDebugProps {
  debugData: DebugData | null;
  onClose: () => void;
}

const AlgorithmDebug: React.FC<AlgorithmDebugProps> = ({ debugData, onClose }) => {
  const [selectedStep, setSelectedStep] = useState<number>(0);
  const [viewMode, setViewMode] = useState<'overview' | 'grid' | 'decisions' | 'integrated'>('overview');
  const [selectedCell, setSelectedCell] = useState<{row: number, col: number} | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number} | null>(null);

  if (!debugData) {
    return null;
  }

  const { explored_nodes, decision_points, grid_exploration } = debugData;

  const renderGridVisualization = () => {
    const { shape, explored, in_path, g_scores, f_scores } = grid_exploration;
    const [rows, cols] = shape;
    
    // Debug: Count in_path cells
    const inPathCount = in_path.flat().filter(Boolean).length;
    console.log(`Grid view - in_path cells: ${inPathCount}`);
    
    // Sample the grid for visualization (too large to render every cell)
    const sampleRate = Math.max(1, Math.floor(Math.max(rows, cols) / 100));
    
    return (
      <div className="grid-visualization">
        <div className="grid-controls">
          <label>
            Sample Rate: 1 in {sampleRate} cells
          </label>
        </div>
        <div 
          className="debug-grid" 
          style={{
            gridTemplateColumns: `repeat(${Math.ceil(cols / sampleRate)}, 8px)`,
            gridTemplateRows: `repeat(${Math.ceil(rows / sampleRate)}, 8px)`
          }}
        >
          {Array.from({ length: Math.ceil(rows / sampleRate) }, (_, r) =>
            Array.from({ length: Math.ceil(cols / sampleRate) }, (_, c) => {
              const actualRow = r * sampleRate;
              const actualCol = c * sampleRate;
              
              if (actualRow >= rows || actualCol >= cols) return null;
              
              const isExplored = explored[actualRow]?.[actualCol];
              const isInPath = in_path[actualRow]?.[actualCol];
              const gScore = g_scores[actualRow]?.[actualCol];
              
              let className = 'grid-cell';
              if (isInPath) className += ' in-path';
              else if (isExplored) className += ' explored';
              
              return (
                <div
                  key={`${r}-${c}`}
                  className={className}
                  title={`Row: ${actualRow}, Col: ${actualCol}, G-Score: ${gScore?.toFixed(2) || 'inf'}`}
                />
              );
            })
          )}
        </div>
      </div>
    );
  };

  // Helper function to find decision at a specific cell
  const getDecisionAtCell = (row: number, col: number): DecisionPoint | null => {
    return decision_points.find(dp => 
      dp.current_node.row === row && dp.current_node.col === col
    ) || null;
  };

  // Helper function to find if a cell was evaluated as a neighbor
  const getNeighborEvaluations = (row: number, col: number): Array<{
    fromDecision: DecisionPoint;
    evaluation: NeighborEvaluation;
  }> => {
    const evaluations: Array<{fromDecision: DecisionPoint; evaluation: NeighborEvaluation}> = [];
    
    decision_points.forEach(dp => {
      const neighborEval = dp.neighbors_evaluated.find(
        n => n.row === row && n.col === col
      );
      if (neighborEval) {
        evaluations.push({ fromDecision: dp, evaluation: neighborEval });
      }
    });
    
    return evaluations;
  };

  const renderIntegratedView = () => {
    const { shape, explored, in_path, g_scores, f_scores } = grid_exploration;
    const [rows, cols] = shape;
    const sampleRate = Math.max(1, Math.floor(Math.max(rows, cols) / 80)); // Finer resolution for integrated view
    
    // Debug: Count in_path cells
    const inPathCount = in_path.flat().filter(Boolean).length;
    console.log(`Debug: Total in_path cells: ${inPathCount}, Grid: ${rows}x${cols}, Sample rate: ${sampleRate}`);
    
    // Find actual in_path cell indices for debugging
    if (inPathCount > 0) {
      const pathIndices: [number, number][] = [];
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (in_path[i]?.[j]) {
            pathIndices.push([i, j]);
          }
        }
      }
      console.log('First 5 in_path cells:', pathIndices.slice(0, 5));
    }
    
    // Get cell info for selected or hovered cell
    const activeCellInfo = selectedCell || hoveredCell;
    let cellDecision: DecisionPoint | null = null;
    let cellEvaluations: Array<{fromDecision: DecisionPoint; evaluation: NeighborEvaluation}> = [];
    
    if (activeCellInfo) {
      cellDecision = getDecisionAtCell(activeCellInfo.row, activeCellInfo.col);
      cellEvaluations = getNeighborEvaluations(activeCellInfo.row, activeCellInfo.col);
    }
    
    return (
      <div className="integrated-view">
        <div className="integrated-grid-section">
          <div className="grid-controls">
            <label>Grid Resolution: 1 in {sampleRate} cells</label>
            {activeCellInfo && (
              <span className="cell-coords">
                Selected: ({activeCellInfo.row}, {activeCellInfo.col})
              </span>
            )}
          </div>
          
          <div 
            className="integrated-grid" 
            style={{
              gridTemplateColumns: `repeat(${Math.ceil(cols / sampleRate)}, 12px)`,
              gridTemplateRows: `repeat(${Math.ceil(rows / sampleRate)}, 12px)`
            }}
          >
            {Array.from({ length: Math.ceil(rows / sampleRate) }, (_, r) =>
              Array.from({ length: Math.ceil(cols / sampleRate) }, (_, c) => {
                const actualRow = r * sampleRate;
                const actualCol = c * sampleRate;
                
                if (actualRow >= rows || actualCol >= cols) return null;
                
                const isExplored = explored[actualRow]?.[actualCol];
                const isInPath = in_path[actualRow]?.[actualCol];
                const gScore = g_scores[actualRow]?.[actualCol];
                const fScore = f_scores[actualRow]?.[actualCol];
                const hasDecision = getDecisionAtCell(actualRow, actualCol) !== null;
                const wasEvaluated = getNeighborEvaluations(actualRow, actualCol).length > 0;
                
                let className = 'integrated-cell';
                if (isInPath) className += ' in-path';
                else if (isExplored) className += ' explored';
                if (hasDecision) className += ' has-decision';
                if (wasEvaluated && !isInPath) className += ' was-evaluated';
                
                if (selectedCell?.row === actualRow && selectedCell?.col === actualCol) {
                  className += ' selected';
                }
                if (hoveredCell?.row === actualRow && hoveredCell?.col === actualCol) {
                  className += ' hovered';
                }
                
                return (
                  <div
                    key={`${r}-${c}`}
                    className={className}
                    onClick={() => setSelectedCell({ row: actualRow, col: actualCol })}
                    onMouseEnter={() => setHoveredCell({ row: actualRow, col: actualCol })}
                    onMouseLeave={() => setHoveredCell(null)}
                    title={`(${actualRow}, ${actualCol}) - G: ${gScore?.toFixed(1) || 'inf'}, F: ${fScore?.toFixed(1) || 'inf'}`}
                  >
                    {hasDecision && <div className="decision-marker">●</div>}
                  </div>
                );
              })
            )}
          </div>
          
          <div className="grid-legend">
            <div className="legend-item">
              <div className="legend-color in-path"></div>
              <span>Final Path</span>
            </div>
            <div className="legend-item">
              <div className="legend-color explored"></div>
              <span>Explored</span>
            </div>
            <div className="legend-item">
              <div className="legend-color was-evaluated"></div>
              <span>Evaluated but not chosen</span>
            </div>
            <div className="legend-item">
              <div className="legend-marker">●</div>
              <span>Decision Point</span>
            </div>
          </div>
        </div>
        
        <div className="integrated-details-section">
          {activeCellInfo && (
            <>
              {cellDecision && (
                <div className="cell-decision-info">
                  <h4>Decision Made at This Cell (Step {cellDecision.step})</h4>
                  <div className="decision-summary">
                    <div>Coordinates: {cellDecision.current_node.lat_lon[0].toFixed(6)}, {cellDecision.current_node.lat_lon[1].toFixed(6)}</div>
                    <div>Evaluated {cellDecision.neighbors_evaluated.length} neighbors</div>
                    <div>Chose {cellDecision.chosen_neighbors.length} improvement(s)</div>
                  </div>
                  
                  <h5>Choices Made:</h5>
                  <div className="mini-neighbors">
                    {cellDecision.chosen_neighbors.map((neighbor, idx) => (
                      <div key={idx} className="mini-neighbor chosen">
                        <strong>{neighbor.direction_name}</strong>
                        <div>Slope: {neighbor.slope_degrees.toFixed(1)}°</div>
                        <div>F-Score: {neighbor.f_score_breakdown.f_score.toFixed(1)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {cellEvaluations.length > 0 && (
                <div className="cell-evaluation-info">
                  <h4>This Cell Was Evaluated {cellEvaluations.length} Time(s)</h4>
                  {cellEvaluations.map((evalInfo, idx) => (
                    <div key={idx} className="evaluation-detail">
                      <h5>From Step {evalInfo.fromDecision.step} at ({evalInfo.fromDecision.current_node.row}, {evalInfo.fromDecision.current_node.col})</h5>
                      <div className="evaluation-summary">
                        <div>Direction: {evalInfo.evaluation.direction_name}</div>
                        <div>Distance: {evalInfo.evaluation.distance_meters.toFixed(1)}m</div>
                        <div>Elevation Δ: {evalInfo.evaluation.elevation_change_m > 0 ? '+' : ''}{evalInfo.evaluation.elevation_change_m.toFixed(1)}m</div>
                        <div>Slope: {evalInfo.evaluation.slope_degrees.toFixed(1)}°</div>
                        <div className={evalInfo.evaluation.is_improvement ? 'chosen-text' : 'rejected-text'}>
                          {evalInfo.evaluation.is_improvement ? '✓ Chosen' : '✗ Rejected'}
                        </div>
                        {!evalInfo.evaluation.is_improvement && (
                          <div className="rejection-reason">
                            G-Score: {evalInfo.evaluation.g_score_breakdown.tentative_g_score.toFixed(1)} 
                            {evalInfo.evaluation.current_g_score < Infinity && 
                              ` (current: ${evalInfo.evaluation.current_g_score.toFixed(1)})`
                            }
                          </div>
                        )}
                      </div>
                      {evalInfo.evaluation.terrain_breakdown.is_obstacle && (
                        <div className="obstacle-info">⚠️ Obstacle detected!</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
              
              {!cellDecision && cellEvaluations.length === 0 && (
                <div className="no-cell-info">
                  <p>This cell was not a decision point and was not evaluated as a neighbor.</p>
                  {explored[activeCellInfo.row]?.[activeCellInfo.col] && (
                    <p>It was explored during the search.</p>
                  )}
                  {in_path[activeCellInfo.row]?.[activeCellInfo.col] && (
                    <p className="in-path-text">It is part of the final path!</p>
                  )}
                </div>
              )}
            </>
          )}
          
          {!activeCellInfo && (
            <div className="help-text">
              <p>Click on any cell in the grid to see:</p>
              <ul>
                <li>Decisions made at that location</li>
                <li>Times it was evaluated as a potential next step</li>
                <li>Why it was chosen or rejected</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderDecisionPoint = (decision: DecisionPoint) => {
    return (
      <div key={decision.step} className="decision-point">
        <h4>Step {decision.step}</h4>
        <div className="current-node">
          <strong>Current Node:</strong> ({decision.current_node.row}, {decision.current_node.col})
          <br />
          <strong>Coordinates:</strong> {decision.current_node.lat_lon[0].toFixed(6)}, {decision.current_node.lat_lon[1].toFixed(6)}
        </div>
        
        <div className="neighbors">
          <h5>Neighbors Evaluated ({decision.neighbors_evaluated.length}):</h5>
          <div className="neighbors-grid">
            {decision.neighbors_evaluated.map((neighbor, idx) => (
              <div 
                key={idx} 
                className={`neighbor ${neighbor.is_improvement ? 'improvement' : 'no-improvement'}`}
              >
                <div className="neighbor-header">
                  <div className="neighbor-pos">
                    ({neighbor.row}, {neighbor.col}) - {neighbor.direction_name}
                  </div>
                  {neighbor.is_improvement && <div className="improvement-indicator">✓ Chosen</div>}
                </div>
                
                <div className="neighbor-details">
                  <div className="terrain-info">
                    <strong>Terrain:</strong>
                    <div>Distance: {neighbor.distance_meters.toFixed(1)}m</div>
                    <div>Elevation Δ: {neighbor.elevation_change_m > 0 ? '+' : ''}{neighbor.elevation_change_m.toFixed(1)}m</div>
                    <div>Slope: {neighbor.slope_degrees.toFixed(1)}°</div>
                    {neighbor.terrain_breakdown.is_obstacle && <div className="obstacle-warning">⚠️ Obstacle</div>}
                  </div>
                  
                  <div className="cost-breakdown">
                    <strong>Cost Breakdown:</strong>
                    <div>Base: {neighbor.terrain_breakdown.base_cost.toFixed(2)}</div>
                    <div>Slope penalty: +{neighbor.terrain_breakdown.slope_penalty.toFixed(2)}</div>
                    <div>Total terrain: {neighbor.terrain_breakdown.total_terrain_cost.toFixed(2)}</div>
                    <div className="formula">{neighbor.cost_breakdown.explanation}</div>
                  </div>
                  
                  <div className="score-breakdown">
                    <strong>G-Score:</strong>
                    <div className="formula">{neighbor.g_score_breakdown.explanation}</div>
                    
                    <strong>F-Score:</strong>
                    <div className="formula">{neighbor.f_score_breakdown.explanation}</div>
                    
                    <div className="score-summary">
                      G: {neighbor.g_score_breakdown.tentative_g_score.toFixed(2)} | 
                      H: {neighbor.f_score_breakdown.h_score.toFixed(2)} | 
                      F: {neighbor.f_score_breakdown.f_score.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="algorithm-debug-overlay">
      <div className="debug-panel">
        <div className="debug-header">
          <h3>A* Algorithm Debug Visualization</h3>
          <button onClick={onClose} className="close-button">×</button>
        </div>

        <div className="debug-controls">
          <div className="view-modes">
            <button 
              className={viewMode === 'overview' ? 'active' : ''}
              onClick={() => setViewMode('overview')}
            >
              Overview
            </button>
            <button 
              className={viewMode === 'grid' ? 'active' : ''}
              onClick={() => setViewMode('grid')}
            >
              Grid Exploration
            </button>
            <button 
              className={viewMode === 'decisions' ? 'active' : ''}
              onClick={() => setViewMode('decisions')}
            >
              Decision Points
            </button>
            <button 
              className={viewMode === 'integrated' ? 'active' : ''}
              onClick={() => setViewMode('integrated')}
            >
              Interactive Grid
            </button>
          </div>

          {viewMode === 'decisions' && (
            <div className="step-controls">
              <label>
                Step: 
                <input
                  type="range"
                  min="0"
                  max={decision_points.length - 1}
                  value={selectedStep}
                  onChange={(e) => setSelectedStep(parseInt(e.target.value))}
                />
                {selectedStep + 1} / {decision_points.length}
              </label>
            </div>
          )}
        </div>

        <div className="debug-content">
          {viewMode === 'overview' && (
            <div className="overview">
              <div className="stats">
                <div className="stat">
                  <strong>Total Nodes Explored:</strong> {explored_nodes.length}
                </div>
                <div className="stat">
                  <strong>Decision Points:</strong> {decision_points.length}
                </div>
                <div className="stat">
                  <strong>Grid Size:</strong> {grid_exploration.shape[0]} × {grid_exploration.shape[1]}
                </div>
                <div className="stat">
                  <strong>Cells Explored:</strong> {
                    grid_exploration.explored.flat().filter(Boolean).length
                  } / {grid_exploration.shape[0] * grid_exploration.shape[1]}
                </div>
              </div>
              
              <div className="algorithm-summary">
                <h4>Algorithm Process:</h4>
                <ol>
                  <li>A* explores nodes by lowest f-score (g-score + heuristic)</li>
                  <li>For each node, it evaluates all 8 neighbors</li>
                  <li>Neighbors with lower costs are added to the search queue</li>
                  <li>The process continues until the destination is reached</li>
                </ol>
              </div>
            </div>
          )}

          {viewMode === 'grid' && renderGridVisualization()}

          {viewMode === 'decisions' && decision_points[selectedStep] && (
            <div className="decisions-view">
              {renderDecisionPoint(decision_points[selectedStep])}
            </div>
          )}

          {viewMode === 'integrated' && renderIntegratedView()}
        </div>
      </div>
    </div>
  );
};

export default AlgorithmDebug;