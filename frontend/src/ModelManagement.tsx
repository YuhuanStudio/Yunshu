import { t } from "./i18n/index.ts";
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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
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
  disabledReason,
  perform,
}: {
  connection: Connection;
  modelId?: string;
  disabled: boolean;
  /** Why the controls are disabled; shown as a tooltip on the import button. */
  disabledReason?: string;
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
              aria-label={t("models.manage.moreActions", { id: modelId })}
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
              {t("models.manage.createAlias")}
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
              {t("models.manage.deleteModel")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <span tabIndex={busy && disabledReason ? 0 : undefined}>
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
                  {t("models.manage.import")}
                </Button>
              </span>
            </TooltipTrigger>
            {busy && disabledReason && (
              <TooltipContent>{disabledReason}</TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      )}

      <Dialog
        open={operation === "pull"}
        onOpenChange={(open) => !open && close()}
      >
        <DialogContent
          onCloseAutoFocus={restore}
          closeLabel={t("models.manage.pull.close")}
          className="max-w-lg"
        >
          <DialogTitle>{t("models.manage.pull.title")}</DialogTitle>
          <DialogDescription>
            {t("models.manage.pull.description")}
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitPull}>
            <div>
              <label htmlFor="model-repository" className="text-sm">
                {t("models.manage.pull.repoLabel")}
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
                {t("models.manage.pull.hint")}
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={close}
              >
                {t("models.manage.cancel")}
              </Button>
              <Button type="submit" disabled={busy || !validRepository}>
                {t("models.manage.pull.start")}
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
          closeLabel={t("models.manage.copy.close")}
          className="max-w-lg"
        >
          <DialogTitle>{t("models.manage.copy.title")}</DialogTitle>
          <DialogDescription>
            {t("models.manage.copy.description", { id: modelId ?? "" })}
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitCopy}>
            <div>
              <label htmlFor="model-alias" className="text-sm">
                {t("models.manage.copy.newId")}
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
                {t("models.manage.cancel")}
              </Button>
              <Button
                type="submit"
                disabled={busy || !validAlias || alias === modelId}
              >
                {t("models.manage.createAlias")}
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
          closeLabel={t("models.manage.delete.close")}
          className="max-w-lg"
        >
          <DialogTitle>{t("models.manage.delete.title")}</DialogTitle>
          <DialogDescription>
            {t("models.manage.delete.description", { id: modelId ?? "" })}
          </DialogDescription>
          <form className="space-y-4" onSubmit={submitDelete}>
            <div>
              <label htmlFor="delete-model-confirm" className="text-sm">
                {t("models.manage.delete.confirmLabel")}
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
                {t("models.manage.cancel")}
              </Button>
              <Button
                type="submit"
                variant="destructive"
                disabled={busy || !exactConfirmation}
              >
                {t("models.manage.deleteModel")}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
