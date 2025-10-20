window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    
    // --- ボタンの取得 ---
    const downloadFullDataBtn = document.getElementById('downloadFullDataBtn');
    const downloadTransitionBtn = document.getElementById('downloadTransitionBtn');
    const clearBtn = document.getElementById('clearBtn');

    // --- グリッド設定 ---
    const GRID_SIZE = 8; // 8x8
    let cellSize = 0;
    
    // Canvasのサイズを「正方形」に設定
    const controlsHeight = document.getElementById('controls').offsetHeight;
    const availableHeight = window.innerHeight - controlsHeight;
    const size = Math.min(window.innerWidth - 4, availableHeight - 4);
    
    canvas.width = size;
    canvas.height = size;
    cellSize = canvas.width / GRID_SIZE;

    // グリッド線を描画する関数
    function drawGrid() {
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;
        for (let i = 1; i < GRID_SIZE; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
            ctx.stroke();
        }
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
    }
    drawGrid();

    // 描画用の設定
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let strokeCount = 0;
    
    // --- ★データ格納用の配列 ---
    let drawingData = []; // 全ての生データ
    let cellTransitions = []; // セル移動のデータ
    let lastCellId = -1; // 直前のセルIDを保持
    // -------------------------

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    
    downloadFullDataBtn.addEventListener('click', () => downloadCSV(drawingData, 'unpitsu_data_full.csv'));
    downloadTransitionBtn.addEventListener('click', () => downloadCSV(cellTransitions, 'unpitsu_data_transitions.csv'));
    clearBtn.addEventListener('click', clearCanvas);

    // ----------------------------------------
    // ヘルパー関数 (座標 -> セルID)
    // ----------------------------------------
    function getCellId(pos) {
        const cellX = Math.floor(pos.x / cellSize);
        const cellY = Math.floor(pos.y / cellSize);
        const validCellX = Math.max(0, Math.min(cellX, GRID_SIZE - 1));
        const validCellY = Math.max(0, Math.min(cellY, GRID_SIZE - 1));
        return (validCellY * GRID_SIZE) + validCellX;
    }
    
    function getCanvasPosition(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    // ----------------------------------------
    // 描画とデータ記録の関数
    // ----------------------------------------

    function startDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = true;
        strokeCount++;
        
        const pos = getCanvasPosition(e);
        const cellId = getCellId(pos);
        lastCellId = cellId; // ストローク開始時のセルIDを保存

        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        recordData(e, 'down', cellId); // 生データを記録
    }

    function draw(e) {
        if (!isDrawing || e.pointerType !== 'pen') return;
        
        const pos = getCanvasPosition(e);
        const cellId = getCellId(pos);

        // --- ★セル移動の検出ロジック ---
        if (cellId !== lastCellId) {
            // セルが変わった場合
            
            // 「隣接」（上下左右）しているかチェック
            const current_x = cellId % GRID_SIZE;
            const current_y = Math.floor(cellId / GRID_SIZE);
            const prev_x = lastCellId % GRID_SIZE;
            const prev_y = Math.floor(lastCellId / GRID_SIZE);
            
            const distance = Math.abs(current_x - prev_x) + Math.abs(current_y - prev_y);
            
            if (distance === 1) { // マンハッタン距離が1 (上下左右)
                // セル移動データを記録
                cellTransitions.push({
                    timestamp: e.timeStamp,
                    stroke_id: strokeCount,
                    from_cell: lastCellId,
                    to_cell: cellId
                });
            }
            lastCellId = cellId; // 直前のセルIDを更新
        }
        // ------------------------------

        // Canvasに描画
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        // 生データを記録
        recordData(e, 'move', cellId);
    }

    function stopDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = false;
        lastCellId = -1; // ペンが離れたらリセット
    }

    // 生データを記録する関数
    function recordData(e, eventType, cellId) {
        const pos = getCanvasPosition(e);
        drawingData.push({
            timestamp: e.timeStamp,
            event_type: eventType,
            stroke_id: strokeCount,
            x: pos.x,
            y: pos.y,
            pressure: e.pressure,
            tilt_x: e.tiltX,
            tilt_y: e.tiltY,
            cell_id: cellId
        });
    }

    // ----------------------------------------
    // CSV処理とクリアの関数
    // ----------------------------------------

    function downloadCSV(data, filename) {
        if (data.length === 0) {
            alert("データがありません。");
            return;
        }

        // データの最初の行からヘッダーを自動生成
        const headers = Object.keys(data[0]).join(',');
        
        // CSVのデータ行を作成
        const csvRows = data.map(row => {
            // オブジェクトの値をヘッダーの順序で取り出す
            return Object.values(row).map(value => {
                // 数値の場合は必要に応じて丸める
                if (typeof value === 'number') {
                    // 小数点以下が多いものを丸める (例: pressure)
                    if (value < 1.0 && value > 0.0) return value.toFixed(4);
                    // 座標
                    if (value > GRID_SIZE) return value.toFixed(2); 
                }
                return value;
            }).join(',');
        });

        const csvContent = headers + "\n" + csvRows.join("\n");
        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function clearCanvas() {
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            cellTransitions = []; // ★移動データもクリア
            strokeCount = 0;
            lastCellId = -1;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid(); // グリッド線を再描画
        }
    }
});