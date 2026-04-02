import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the drug search workspace", () => {
  render(<App />);

  expect(
    screen.getByRole("heading", { name: /graphrxinsight search/i })
  ).toBeInTheDocument();
  expect(screen.getByLabelText(/drug 1/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/drug 2/i)).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /check interaction/i })
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /observed interaction/i })
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /no interaction observed/i })
  ).toBeInTheDocument();
});
