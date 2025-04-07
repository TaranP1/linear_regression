import { Button } from "@/components/ui/button";

interface ButtonProps {
  onClick: () => void;
}

export default function ButtonComponent({ onClick }: ButtonProps) {
  return (
    <Button onClick={onClick}>
      Visualize Regression
    </Button>
  );
}
