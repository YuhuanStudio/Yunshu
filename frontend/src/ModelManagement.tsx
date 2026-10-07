import { useRef, useState } from "react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  Input,
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@yuhuanowo/yunui";
import { MoreHorizontal, Copy, Trash2 } from "lucide-react";
import type { Connection } from "./api";
import { copyModel, deleteModel, pullModel } from "./management-api";
import type { Perform } from "./ModelActions";

type Operation = "pull" | "copy" | "delete" | null;

export function ModelManagement({
  connection,
  modelId,
  disabled,
  perform,
}: {
  connection: Connection;
  modelId?: string;
  disabled: boolean;
  perform: Perform;
}) {
  const [operation, setOperation] = useState<Operation>(null);
  const [repository, setRepository] = useState("");
  const [alias, setAlias] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const busy = disabled;
  const opener = useRef<HTMLButtonElement | null>(null);
  const restore = (event: Event) => {
    if (opener.current?.isConnected) {
      event.preventDefault();
      opener.current.focus({ preventScroll: true });
    }
  };
  const close = () => {
    setOperation(null);
    setConfirmation("");
  };

  const validRepository = /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(
    repository,
  );
  const validAlias =
    alias.length > 0 &&
    alias === alias.trim() &&
    !alias.includes("\\") &&
    !alias.includes("\0") &&
    alias
      .split("/")
      .every((part) => part !== "" && part !== "." && part !== "..");
  const exactConfirmation = modelId !== undefined && confirmation === modelId;

  function submitPull(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!validRepository || busy) return;
    const value = repository;
    close();
    void perform(`pull:${value}`, () => pullModel(connection, value));
  }

  function submitCopy(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!modelId || !validAlias || alias === modelId || busy) return;
    const destination = alias;
    close();
    void perform(`copy:${modelId}`, () =>
      copyModel(connection, modelId, destination),
    );
  }

  function submitDelete(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!modelId || !exactConfirmation || busy) return;
    const value = modelId;
    close();
    void perform(`delete:${value}`, () => deleteModel(connection, value));
  }

  return (
    <>
      {modelId ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              ref={opener}
              variant="ghost"
              size="sm"
              disabled={busy}
              aria-label={`${modelId} 的更多操作`}
            >
              <MoreHorizontal size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            onCloseAutoFocus={(event) => {
              if (operation) event.preventDefault();
            }}
          >
            <DropdownMenuItem
              onSelect={() => {
                setAlias("");
                setOperation("copy");
              }}
            >
              <Copy size={14} />
              建立別名
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-error"
              onSelect={() => {
                setConfirmation("");
                setOperation("delete");
              }}
            >
              <Trash2 size={14} />
              刪除模型
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <Button
          variant="secondary"
          size="sm"
          disabled={busy}
          onClick={(event) => {
            opener.current = event.currentTarget;
            setRepository("");
            setOperation("pull");
          }}
        >
          匯入模型
        </Button>
      )}

      <Dialog
        open={operation === "pull"}
        onOpenChange={(open) => !open && close()}
      >
        <DialogContent
          onCloseAutoFocus={restore}
          closeLabel="關閉匯入模型"
          className="max-w-lg"
        >
          <DialogTitle>匯入 Hugging Face MLX 模型</DialogTitle>
          <DialogDescription>
            輸入原生 MLX safetensors 儲存庫
            ID（org/name）。這會開始實際下載；檔案大小未知，服務不提供下載進度或取消操作。
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitPull}>
            <div>
              <label htmlFor="model-repository" className="text-sm">
                儲存庫 ID
              </label>
              <Input
                id="model-repository"
                className="mt-2 font-mono"
                value={repository}
                onChange={(event) => setRepository(event.target.value)}
                placeholder="mlx-community/model-name"
                autoComplete="off"
                required
              />
              <p className="mt-2 text-xs text-muted-foreground">
                僅支援 Hugging Face 原生 MLX 模型；Ollama registry 與 GGUF
                不支援。
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={close}
              >
                取消
              </Button>
              <Button type="submit" disabled={busy || !validRepository}>
                開始下載
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={operation === "copy"}
        onOpenChange={(open) => !open && close()}
      >
        <DialogContent
          onCloseAutoFocus={restore}
          closeLabel="關閉建立別名"
          className="max-w-lg"
        >
          <DialogTitle>建立模型別名</DialogTitle>
          <DialogDescription>
            為 {modelId}{" "}
            建立持久別名。服務會建立指向相同權重的連結，不會複製模型檔案。
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitCopy}>
            <div>
              <label htmlFor="model-alias" className="text-sm">
                新模型 ID
              </label>
              <Input
                id="model-alias"
                className="mt-2 font-mono"
                value={alias}
                onChange={(event) => setAlias(event.target.value)}
                placeholder="team/model-alias"
                autoComplete="off"
                required
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={close}
              >
                取消
              </Button>
              <Button
                type="submit"
                disabled={busy || !validAlias || alias === modelId}
              >
                建立別名
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={operation === "delete"}
        onOpenChange={(open) => !open && close()}
      >
        <DialogContent
          onCloseAutoFocus={restore}
          closeLabel="關閉刪除模型確認"
          className="max-w-lg"
        >
          <DialogTitle>刪除模型與權重？</DialogTitle>
          <DialogDescription>
            這項操作可能永久移除模型權重，無法復原。請輸入完整模型 ID「{modelId}
            」確認。
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitDelete}>
            <div>
              <label htmlFor="delete-model-confirm" className="text-sm">
                確認模型 ID
              </label>
              <Input
                id="delete-model-confirm"
                className="mt-2 font-mono"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                autoComplete="off"
                required
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={close}
              >
                取消
              </Button>
              <Button
                type="submit"
                variant="destructive"
                disabled={busy || !exactConfirmation}
              >
                刪除模型
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
